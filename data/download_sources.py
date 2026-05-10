#!/usr/bin/env python3
"""Clone the 45 reference web app source repos for VFS Stage 5.

The canonical (app_id, github_url, license) list lives on HuggingFace at
    https://huggingface.co/datasets/ui2app-anon/UI2App/blob/main/reference_apps_manifest.csv
plus the same URLs (by repo basename) in
    https://huggingface.co/datasets/ui2app-anon/UI2App/blob/main/croissant_rai_UI2App.json
under `prov:wasDerivedFrom` (NeurIPS 2026 D&B RAI metadata).

This script fetches the CSV at runtime — the code repo intentionally ships no
copy of the dataset's URL list, so the dataset stays the single source of truth.

Output: <dest>/<app_id>/  (clean project root, ready for `pnpm install`)

Idempotent: existing target dirs are skipped. Failures are logged to
`data/failed.log` without aborting the rest.

Usage:
    python data/download_sources.py                       # all 45 → data/sources/
    python data/download_sources.py --dest /scratch/src   # custom destination
    python data/download_sources.py --app 19_shadcn-admin # one project
    python data/download_sources.py --dry-run             # print commands only
"""
import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

HF_DATASET_REPO = "ui2app-anon/UI2App"
HF_MANIFEST_FILE = "reference_apps_manifest.csv"

HERE = Path(__file__).resolve().parent
FAILED_LOG = HERE / "failed.log"

# Monorepo subdirectories — packaging-side concern (which subfolder of the
# cloned repo is the actual app), not dataset metadata. Hardcoded here so the
# CSV on HF stays a clean attribution record.
SUBDIR_OVERRIDES = {
    "27_shadcnstore-dashboard": "nextjs-version",
    "67_shadboard":             "full-kit",
    "68_refine-finefoods":      "examples/finefoods-antd",
}


def fetch_manifest():
    """Download reference_apps_manifest.csv from HF and return rows."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
    try:
        path = hf_hub_download(repo_id=HF_DATASET_REPO, filename=HF_MANIFEST_FILE,
                               repo_type="dataset")
    except Exception as e:
        sys.exit(
            f"ERROR: failed to download {HF_MANIFEST_FILE} from "
            f"https://huggingface.co/datasets/{HF_DATASET_REPO}\n"
            f"  Reason: {str(e).splitlines()[0]}\n"
            f"  Check: (a) network, (b) HF_DATASET_REPO matches the published "
            f"dataset, (c) `huggingface-cli login` if it's gated."
        )
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def clone_one(row, dest_root, dry_run):
    app_id = row["app_id"]
    url = row["github_url"]
    subdir = SUBDIR_OVERRIDES.get(app_id, "")
    target = dest_root / app_id

    if target.exists():
        print(f"  SKIP   {app_id:38s} (already at {target})")
        return "skip"

    if dry_run:
        if subdir:
            print(f"  DRYRUN {app_id:38s} clone {url} → temp, then move {subdir}/ → {target}")
        else:
            print(f"  DRYRUN {app_id:38s} clone {url} → {target}")
        return "dryrun"

    if subdir:
        tmp = dest_root / f"{app_id}__tmp_clone"
        if tmp.exists():
            shutil.rmtree(tmp)
        cmd = ["git", "clone", "--depth", "1", url, str(tmp)]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=600)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            err = (e.stderr.decode()[:300] if hasattr(e, "stderr") and e.stderr else str(e))
            print(f"  FAIL   {app_id:38s} {url} — {err.strip()[:120]}")
            shutil.rmtree(tmp, ignore_errors=True)
            return f"fail: {err.strip()}"
        sub_path = tmp / subdir
        if not sub_path.is_dir():
            print(f"  FAIL   {app_id:38s} subdir '{subdir}' not found in cloned repo")
            shutil.rmtree(tmp, ignore_errors=True)
            return f"fail: subdir '{subdir}' missing"
        shutil.move(str(sub_path), str(target))
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"  OK     {app_id:38s} {url} (subdir: {subdir})")
        return "ok"

    cmd = ["git", "clone", "--depth", "1", url, str(target)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        err = (e.stderr.decode()[:300] if hasattr(e, "stderr") and e.stderr else str(e))
        print(f"  FAIL   {app_id:38s} {url} — {err.strip()[:120]}")
        shutil.rmtree(target, ignore_errors=True)
        return f"fail: {err.strip()}"
    print(f"  OK     {app_id:38s} {url}")
    return "ok"


def main():
    parser = argparse.ArgumentParser(description="Clone source repos for VFS Stage 5")
    parser.add_argument("--dest", default=str(HERE / "sources"),
                        help="Destination directory (default: data/sources/)")
    parser.add_argument("--app", help="Only clone this app_id")
    parser.add_argument("--dry-run", action="store_true", help="Print git commands without executing")
    args = parser.parse_args()

    print(f"Fetching {HF_MANIFEST_FILE} from {HF_DATASET_REPO}…")
    rows = fetch_manifest()
    if args.app:
        rows = [r for r in rows if r["app_id"] == args.app]
        if not rows:
            sys.exit(f"app_id '{args.app}' not in {HF_MANIFEST_FILE}")

    dest_root = Path(args.dest).resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    print(f"\nCloning {len(rows)} repo(s) to {dest_root} (dry_run={args.dry_run})\n")

    results = {"ok": 0, "skip": 0, "dryrun": 0, "fail": 0}
    failures = []
    for row in rows:
        outcome = clone_one(row, dest_root, args.dry_run)
        if outcome.startswith("fail"):
            results["fail"] += 1
            failures.append((row["app_id"], row["github_url"], outcome))
        else:
            results[outcome] = results.get(outcome, 0) + 1

    print(f"\nSummary: {results['ok']} cloned, {results['skip']} skipped, "
          f"{results['dryrun']} dry-run, {results['fail']} failed")

    if failures:
        with open(FAILED_LOG, "a") as f:
            for app_id, url, err in failures:
                f.write(f"{app_id}\t{url}\t{err}\n")
        print(f"Failures written to {FAILED_LOG}")
        sys.exit(1)


if __name__ == "__main__":
    main()
