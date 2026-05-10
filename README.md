# UI2App: Benchmark Evaluation Pipeline

Anonymized code release for the UI2App benchmark.

## Quickstart

```bash
pip install -r requirements.txt
playwright install chromium
brew install tesseract             # macOS;  Linux: sudo apt install tesseract-ocr
cp .env.example .env               # fill in your API key + base_url
```

## 1. Get the dataset (screenshots + manifests, ~50 MB)

```bash
python data/sync_dataset.py                       # all 45 apps → data/apps/
python data/sync_dataset.py --app <app_id>        # one app
```

## 2. (Optional) Get source repos for full DOM-based VFS (~5-15 GB)

```bash
python data/download_sources.py                   # all 45 → data/sources/
python data/download_sources.py --app <app_id>    # one app
```

If skipped, Stage 5 falls back to OCR on input screenshots — real numbers,
less precise than DOM.

## 3. Run the pipeline (single app, Stage 1-6)

```bash
python benchmark/pipeline.py --project <app_id> --model <model_name>
```

Output: `data/runs/<app_id>/<model>/run_<ts>/{result.json, generated/, …}`.

For batch runs across all apps: `python benchmark/run_all.py --model <model_name> --concurrency 5`.

To re-score an existing run without regenerating:

```bash
python benchmark/pipeline.py --skip-gen --run-dir <path-to-existing-run>
```

## Layout

```
UI2App-Code/
├── benchmark/         # pipeline code (Stage 1-6)
│   ├── pipeline.py    # entry: run_pipeline + run_evaluation
│   ├── run_all.py     # batch runner
│   ├── stages/        # generate / coverage / visual
│   └── support/       # auth / blocks / server / projects / llm
└── data/              # all dataset I/O (gitignored except scripts)
    ├── sync_dataset.py
    ├── download_sources.py
    ├── apps/          # screenshots + manifests (created by sync_dataset.py)
    ├── sources/       # cloned source repos (created by download_sources.py)
    └── runs/          # pipeline output
```

## Dataset

Hosted separately: <https://huggingface.co/datasets/ui2app-anon/UI2App>

## License

MIT (this repository); CC-BY-4.0 (dataset).
