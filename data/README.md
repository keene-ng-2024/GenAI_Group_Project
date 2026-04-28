# Dataset Preparation and Reviewer Agreement Metrics

## Overview
This directory contains the prepared dataset splits and baseline metrics for the AI Research Paper Critique project, fulfilling the Phase 1 responsibilities of the Data & Dataset Lead.

## Setup

### 1. Obtain the raw data
Download `ReviewCritique.jsonl` from the [ReviewCritique dataset repository](https://github.com/jiangshdd/ReviewCritique/tree/main/data) and place it at:
```
data/ReviewCritique.jsonl
```

### 2. Parse reviews
Run the parser to produce `data/processed/reviews_parsed.json`:
```bash
python -m src.data_processing.parse_reviews
```

### 3. Build ground-truth critique dictionaries
Run the dict builder to produce per-paper ground-truth files in `data/processed/critique_dicts/`:
```bash
python -m src.data_processing.build_critique_dict
```

After these steps, the pipeline is ready to run. See the root `README.md` for platform-specific run commands.

