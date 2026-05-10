#!/usr/bin/env python3
"""Pull screenshots + manifests from the UI2App HuggingFace dataset into
data/apps/<app_id>/ — the layout the pipeline expects.

The HF dataset stores files at `screenshots/<app_id>/{manifest.json, img_*.png}`;
this script copies them to `data/apps/<app_id>/` (flat, no `screenshots/` prefix)
so `support.projects.discover_projects()` can find them directly.

Usage:
    python data/sync_dataset.py                       # pull all 45 apps
    python data/sync_dataset.py --app 19_shadcn-admin # pull one
    python data/sync_dataset.py --apps 19 39 45       # pull a few (id prefixes)
"""
import argparse
import shutil
import sys
from pathlib import Path

HF_DATASET_REPO = "ui2app-anon/UI2App"
HERE = Path(__file__).resolve().parent
APPS_DIR = HERE / "apps"


def main():
    parser = argparse.ArgumentParser(description="Sync UI2App screenshots from HuggingFace")
    parser.add_argument("--app", help="Pull a single app_id")
    parser.add_argument("--apps", nargs="*", help="Pull apps whose id starts with any of these prefixes")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download, HfApi
    except ImportError:
        sys.exit("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")

    api = HfApi()
    try:
        all_files = api.list_repo_files(HF_DATASET_REPO, repo_type="dataset")
    except Exception as e:
        sys.exit(f"ERROR: failed to list {HF_DATASET_REPO} on HuggingFace: {e}")

    discovered = {f.split("/")[1] for f in all_files
                  if f.startswith("screenshots/") and f.count("/") >= 2}

    if args.app:
        target_apps = [args.app] if args.app in discovered else []
        if not target_apps:
            sys.exit(f"app_id '{args.app}' not in dataset. Available: {sorted(discovered)[:5]}…")
    elif args.apps:
        target_apps = sorted([a for a in discovered
                              if any(a.startswith(p) for p in args.apps)])
        if not target_apps:
            sys.exit(f"No apps match prefixes {args.apps}.")
    else:
        target_apps = sorted(discovered)

    patterns = [f"screenshots/{a}/*" for a in target_apps]
    print(f"Pulling {len(target_apps)} app(s) from {HF_DATASET_REPO}…")
    snap_path = Path(snapshot_download(repo_id=HF_DATASET_REPO, repo_type="dataset",
                                       allow_patterns=patterns))

    APPS_DIR.mkdir(parents=True, exist_ok=True)
    for app in target_apps:
        src = snap_path / "screenshots" / app
        dst = APPS_DIR / app
        if not src.is_dir():
            print(f"  SKIP {app} (not found in snapshot)")
            continue
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        for f in src.iterdir():
            shutil.copy2(f, dst / f.name)
        n_imgs = len(list(dst.glob("img_*.png")))
        print(f"  OK   {app:38s} → {dst.relative_to(HERE.parent)} ({n_imgs} screenshots)")

    print(f"\nDone. Apps now at {APPS_DIR.relative_to(HERE.parent)}/")


if __name__ == "__main__":
    main()
