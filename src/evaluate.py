import os
import json
import warnings
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix, matthews_corrcoef, cohen_kappa_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import nltk
from nltk.translate import meteor_score
from nltk.tokenize import word_tokenize
from bert_score import score as calculate_bert_score
import csv
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Suppress warnings from huggingface/transformers often triggered by BERTScore
warnings.filterwarnings("ignore")

# Ensure required NLTK data is downloaded for METEOR
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('wordnet', quiet=True)

def evaluate_observer_phase(truth_json_path, pred_json_data, chunk_size=10):
    """
    Evaluates Phase 1: Chunk-by-chunk detection metrics.
    Aggregates the VLM's frame-level predictions into chunks and compares them to the ground truth.
    """
    with open(truth_json_path, 'r', encoding='utf-8') as f:
        truth_data = json.load(f)

    # 1. Extract ground truth (chunk-level)
    y_true = [chunk.get("has_violation", False) for chunk in truth_data]

    # 2. Aggregate VLM predictions into chunks
    y_pred = []
    # Loop through the flat frame predictions in steps of `chunk_size`
    for i in range(0, len(pred_json_data), chunk_size):
        chunk_frames = pred_json_data[i:i + chunk_size]
        
        # A chunk is flagged if ANY frame inside it shows an active violation
        chunk_flagged = any(
            frame.get("propeller_active", False) and frame.get("danger_zone_violation", False) 
            for frame in chunk_frames
        )
        y_pred.append(chunk_flagged)

    if len(y_true) != len(y_pred):
        print(f"Warning: Chunk count mismatch! Truth: {len(y_true)} chunks, Pred: {len(y_pred)} chunks")
        min_len = min(len(y_true), len(y_pred))
        y_true, y_pred = y_true[:min_len], y_pred[:min_len]
        
    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        }

    # Calculate metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel() if len(set(y_true + y_pred)) > 1 else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred) if len(set(y_true + y_pred)) > 1 else 0.0
    kappa = cohen_kappa_score(y_true, y_pred) if len(set(y_true + y_pred)) > 1 else 0.0
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "npv": npv,
        "mcc": mcc,
        "kappa": kappa,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
    }

    # Error analysis: false positives and false negatives (chunk indices)
    false_positives = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if p and not t]
    false_negatives = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t and not p]
    metrics["false_positives"] = false_positives
    metrics["false_negatives"] = false_negatives
    return metrics

def evaluate_analyst_phase(truth_report_path, pred_report_text):
    """
    Evaluates Phase 2: Narrative report generation using BERTScore and METEOR.
    """
    with open(truth_report_path, 'r', encoding='utf-8') as f:
        truth_report_text = f.read()

    # 1. METEOR Score Calculation
    # METEOR requires lists of tokens (words)
    truth_tokens = word_tokenize(truth_report_text)
    pred_tokens = word_tokenize(pred_report_text)
    calc_meteor = meteor_score.single_meteor_score(truth_tokens, pred_tokens)

    # 2. BERTScore Calculation
    P, R, F1 = calculate_bert_score(
        [pred_report_text], 
        [truth_report_text], 
        lang="en", 
        model_type="distilbert-base-uncased"
    )

    # 3. BLEU Score (sentence-level, smoothing)
    smoothie = SmoothingFunction().method4
    bleu = sentence_bleu([truth_tokens], pred_tokens, smoothing_function=smoothie)

    # 4. ROUGE Score
    rouge = Rouge()
    try:
        rouge_scores = rouge.get_scores(pred_report_text, truth_report_text)[0]
    except Exception:
        rouge_scores = {"rouge-1": {"f": 0.0}, "rouge-l": {"f": 0.0}}

    metrics = {
        "meteor_score": calc_meteor,
        "bert_precision": P.item(),
        "bert_recall": R.item(),
        "bert_f1": F1.item(),
        "bleu": bleu,
        "rouge_1_f": rouge_scores.get("rouge-1", {}).get("f", 0.0),
        "rouge_l_f": rouge_scores.get("rouge-l", {}).get("f", 0.0)
    }
    return metrics

def run_evaluation(eval_dir="eval_data"):
    """
    Main loop to run evaluations across all folders in the eval_data directory.
    Note: You will need to import and initialize your SafetyAgent to get `pred_json_data` 
    and `pred_report_text` dynamically, or load them if you pre-generated them.
    """

    print("Starting Automated Evaluation Pipeline...\n")
    for video_folder in os.listdir(eval_dir):
        folder_path = os.path.join(eval_dir, video_folder)
        if not os.path.isdir(folder_path):
            continue
        print(f"--- Evaluating: {video_folder} ---")
        # Define paths based on your structure
        truth_json_path = os.path.join(folder_path, f"{video_folder}_truths.json")
        truth_report_path = os.path.join(folder_path, f"{video_folder}_report.txt")
        pred_json_path = os.path.join(folder_path, f"{video_folder}_pred.json")
        pred_report_path = os.path.join(folder_path, f"{video_folder}_pred_report.txt")

        eval_result = {}

        # Observer phase
        if os.path.exists(truth_json_path) and os.path.exists(pred_json_path):
            with open(pred_json_path, 'r', encoding='utf-8') as f:
                pred_json_data = json.load(f)
            observer_metrics = evaluate_observer_phase(truth_json_path, pred_json_data)
            print("[Observer] Metrics:", {k: v for k, v in observer_metrics.items() if not isinstance(v, list)})
            print("  False Positives (chunks):", observer_metrics["false_positives"])
            print("  False Negatives (chunks):", observer_metrics["false_negatives"])
            eval_result["observer"] = observer_metrics
        else:
            print(f"Missing truth or prediction JSON for {video_folder}")

        # Analyst phase
        if os.path.exists(truth_report_path) and os.path.exists(pred_report_path):
            with open(pred_report_path, 'r', encoding='utf-8') as f:
                pred_report_text = f.read()
            analyst_metrics = evaluate_analyst_phase(truth_report_path, pred_report_text)
            print("[Analyst] Metrics:", analyst_metrics)
            eval_result["analyst"] = analyst_metrics
        else:
            print(f"Missing truth or prediction report for {video_folder}")

        # Save per-video evaluation file
        eval_path = os.path.join(folder_path, f"{video_folder}_eval.json")
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump(eval_result, f, indent=2, cls=NumpyEncoder)
        print(f"Saved evaluation to {eval_path}\n")

if __name__ == "__main__":
    run_evaluation()