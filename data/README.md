# Data

## Directory layout

```
data/
├── raw/                     # original review files — place them here
│   ├── <paper_id>.json      # OpenReview JSON export (preferred)
│   └── <paper_id>.pdf       # or raw PDF
│
└── processed/
    ├── reviews_parsed.json  # output of parse_reviews.py
    └── critique_dicts/
        └── <paper_id>.json  # output of build_critique_dict.py
```

## How to obtain data

### Option A — OpenReview (recommended)

1. Browse to a paper on [openreview.net](https://openreview.net).
2. Use the OpenReview API or the bulk-download tool:
   ```bash
   pip install openreview-py
   python - <<'EOF'
   import openreview, json, pathlib

   client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')
   # Replace with your venue / paper IDs
   notes = client.get_notes(forum='<forum_id>', details='replies')
   pathlib.Path('data/raw/<paper_id>.json').write_text(
       json.dumps([n.to_json() for n in notes], indent=2)
   )
   EOF
   ```

### Option B — Manual PDF

Place peer-review PDFs into `data/raw/` named `<paper_id>.pdf`.  The parser
will extract text with pypdf (lower quality than structured JSON).

## Running the pipeline

```bash
# 1. Parse raw files → reviews_parsed.json
python -m src.data_processing.parse_reviews

# 2. Distil reviews → critique dicts (requires ANTHROPIC_API_KEY)
python -m src.data_processing.build_critique_dict
```

## Notes

- Files in `data/raw/` and `data/processed/` are **gitignored** to avoid
  committing potentially sensitive review content.
- The `critique_dicts/` sub-directory is the ground truth used by the scorer.
