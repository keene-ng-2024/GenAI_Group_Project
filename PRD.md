# PRD: GenAI_Group_Project

## Overview
An SMU GenAI group project (earlier milestone or parallel track to GenAI_FinalGroupProject) comparing single-call LLM baselines against multi-agent critique workflows for peer-review-style AI paper analysis. Evaluates coverage of human-derived critique points using semantic similarity scoring.

## Goals
- Same core comparison as GenAI_FinalGroupProject: single-call vs agentic multi-loop
- Parse human peer reviews to build per-paper critique dictionaries
- Run baseline and multi-agent implementations
- Score semantic similarity recall/precision against ground truth

## Non-Goals
- Production tool
- Real-time review API

## Tech Stack
- **Language**: Python 3.x
- **LLM**: GPT-4.1-mini (OpenAI)
- **Libraries**: `openai`, various AI/ML libraries (see requirements.txt)
- **Platforms**: n8n, Dify (via config.yaml)

## Architecture
```
GenAI_Group_Project/
├── config.yaml           # Model configuration
├── requirements.txt
├── .env.example
├── data/
│   ├── raw/              # Human review source files
│   └── processed/        # Parsed critique dicts
├── results/              # Evaluation outputs
└── src/                  # (if present) Implementation scripts
```

## Deployment / Run
```bash
pip install -r requirements.txt
cp .env.example .env
python src/...
```

## Constraints & Notes
- This repo represents an earlier or parallel version of the GenAI_FinalGroupProject study
- Academic project for SMU course — not production-grade
- Requires OpenAI API key; uses GPT-4.1-mini (low cost)
- See GenAI_FinalGroupProject for the final/refined version of this study
