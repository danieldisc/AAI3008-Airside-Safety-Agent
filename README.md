# AAI3008-Airside-Safety-Agent
Multimodal LLM Agent for Automated Airport Ground Handling Incident Investigation.

Given a ramp/airside video clip, the system:
1. Extracts frame-level safety observations via a VLM (OpenAI or Gemini)
2. Retrieves relevant passages from ground handling manuals (TF-IDF RAG)
3. Maps observations to violation rules with evidence citations
4. Generates coaching/teachable moments for each violation
5. Produces a structured final safety report

## Project Structure

```
src/
  app.py              # Streamlit UI for interactive video analysis
  vlm_agent.py        # VLM backend (OpenAI / Gemini)
  report_gen.py       # PDF report generation
  rag/
    run_pipeline.py      # CLI entry point — runs all 5 pipeline stages
    vlm_incident.py      # Converts VLM output to structured incident payload
    incident_retrieval.py # TF-IDF retrieval over manuals
    llm2_mapper.py       # Maps retrieved claims to violation rules
    llm3_teachable.py    # Generates coaching moments
    llm4_report.py       # Builds final consolidated report
    build_index.py       # (Re)builds the TF-IDF index from manuals/
manuals/              # Source PDFs for the retrieval index
rag_index/            # Persisted TF-IDF index
rule_packs/           # JSON rule definitions for violation mapping
data/                 # Video clips for analysis
outputs/              # Pipeline run outputs (created on first run)
```

## Setup

```powershell
pip install -r requirements.txt
```

Create a `.env` file in the repo root with your API key(s):

```
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
```

## Running the Pipeline

From the repo root (`AAI3008-Airside-Safety-Agent`):

```powershell
cd src
```

Run using video input:

```powershell
python -m rag.run_pipeline --video-path "../data/your_clip.mp4" --vlm-engine OpenAI --output-dir ../outputs/my_run
```

Use `--vlm-engine Gemini` to switch to the Gemini backend.

Run using a pre-built incident JSON (skip VLM extraction):

```powershell
python -m rag.run_pipeline --incident-json "../outputs/smoke_test/incident_from_vlm.json" --output-dir ../outputs/my_run
```

Optional: rebuild the retrieval index before running:

```powershell
python -m rag.build_index --output-dir ../rag_index
python -m rag.run_pipeline --video-path "../data/your_clip.mp4" --rebuild-index --output-dir ../outputs/my_run
```

### Output files

| File | Description |
|---|---|
| `incident_from_vlm.json` | Structured observations extracted from the video |
| `analyst_report.txt` | Human-readable analyst narrative |
| `retrieval_sample.json` | Manual passages retrieved per observation |
| `violations_sample.json` | Violation rules matched with evidence citations |
| `teachable_sample.json` | Coaching points for each violation |
| `report_sample.json` | Final consolidated safety report |

### CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--video-path` | — | Path to video file (required unless `--incident-json` given) |
| `--incident-json` | — | Path to a pre-built incident JSON (skips VLM step) |
| `--vlm-engine` | `Gemini` | VLM backend: `OpenAI` or `Gemini` |
| `--output-dir` | `outputs` | Directory to write results |
| `--top-k` | `3` | Manual chunks retrieved per claim |
| `--rebuild-index` | off | Rebuild the TF-IDF index before running |

## Streamlit UI

```powershell
streamlit run src/app.py
```

Upload a video and run dual-engine analysis (Gemini + OpenAI concurrently) to compare outputs and evaluation metrics.