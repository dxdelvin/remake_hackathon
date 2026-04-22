# ZEISS Smart Energy Assistant

A FastAPI application that analyzes ZEISS microscope workflow telemetry and recommends energy-saving actions using a hybrid rule-based and machine-learning engine.

<img width="1917" height="907" alt="image" src="https://github.com/user-attachments/assets/256a529d-6ddb-4509-8b0a-652e260cf154" />
<img width="1874" height="1025" alt="image" src="https://github.com/user-attachments/assets/fa462a18-f050-4cc0-9b35-54e402d6ff23" />


## Features

- MLP classifier trained on labeled workflow telemetry (scenarios S1–S10)
- Rule-based engine for interpretable, scenario-specific recommendations
- Hybrid engine combining ML predictions with rule outputs
- REST API for uploading CSVs and receiving per-segment energy analysis
- Web dashboard for visualizing results

## Project Structure

```
├── main.py                  # FastAPI entry point
├── requirements.txt
├── app/
│   ├── data_processor.py    # CSV loading and segment aggregation
│   ├── energy_calculator.py # Energy metrics
│   ├── hybrid_engine.py     # Combines ML + rule engine
│   ├── ml_model.py          # MLP model training and evaluation
│   └── rule_engine.py       # Rule-based recommendations
├── data/
│   └── training/            # Labeled scenario CSVs (S1–S13)
└── templates/
    └── index.html           # Dashboard UI
```

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`

## Setup

```bash
pip install -r requirements.txt
```

## Running

```bash
python main.py
```

The server starts at `http://localhost:8000`.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web dashboard |
| `POST` | `/analyze` | Upload a workflow CSV, returns segment analysis JSON |
| `GET` | `/health` | Model readiness probe |

## Training Data

Scenario CSVs follow the naming pattern `S{n}_*_v4.csv` and must contain a `recommended_action` column for supervised training. S13 is held out as a test scenario and evaluated separately at startup.

