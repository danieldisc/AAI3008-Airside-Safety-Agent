import os
import json
import warnings
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import nltk
from nltk.translate import meteor_score
from nltk.tokenize import word_tokenize
from bert_score import score as calculate_bert_score

# Suppress warnings from huggingface/transformers often triggered by BERTScore
warnings.filterwarnings("ignore")

# Ensure required NLTK data is downloaded for METEOR
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')  # NEW: Check for punkt_tab
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)  # NEW: Download punkt_tab
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

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0)
    }
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
    # Computes precision, recall, and F1 based on contextual embeddings
    # Using a fast, standard model for English text
    P, R, F1 = calculate_bert_score(
        [pred_report_text], 
        [truth_report_text], 
        lang="en", 
        model_type="distilbert-base-uncased"
    )

    metrics = {
        "meteor_score": calc_meteor,
        "bert_precision": P.item(),
        "bert_recall": R.item(),
        "bert_f1": F1.item()
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
        
        # --- STOP & GENERATE ---
        # In a fully automated loop, you would call your SafetyAgent here:
        # agent = SafetyAgent()
        # video_path = os.path.join(folder_path, f"{video_folder}.mp4")
        # pred_json_data, pred_report_text = agent.analyze_pipeline_for_eval(video_path)
        
        # For demonstration, we assume you have the predictions ready.
        # You will need to slightly modify `vlm_agent.py` to return the JSON logs 
        # alongside the final text if you want to evaluate both in one go.
        
        if os.path.exists(truth_json_path):
            # observer_metrics = evaluate_observer_phase(truth_json_path, pred_json_data)
            print("[Observer] Metrics calculation ready.")
        else:
            print(f"Missing truth JSON for {video_folder}")
            
        if os.path.exists(truth_report_path):
            # analyst_metrics = evaluate_analyst_phase(truth_report_path, pred_report_text)
            print("[Analyst] Metrics calculation ready.")
        else:
            print(f"Missing truth report for {video_folder}")
            
        print("\n")

if __name__ == "__main__":
    run_evaluation()