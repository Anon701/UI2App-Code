#!/usr/bin/env python3
"""
Run UI2App pipeline on all projects concurrently.

Usage:
  python benchmark/run_all.py                                    # all projects, claude-sonnet-4
  python benchmark/run_all.py --model gpt-4.1 --concurrency 5   # gpt-4.1, 5 at a time
  python benchmark/run_all.py --projects 01 03 09                # specific projects only
"""

import argparse, subprocess, sys, time, json, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE = SCRIPT_DIR / "pipeline.py"

sys.path.insert(0, str(SCRIPT_DIR))
from support.projects import discover_projects, RUNS_DIR
BASE_PORT = 5280


def run_one(project, model, port):
    """Run pipeline for one project. Returns (project, success, result_path, elapsed, error)."""
    start = time.time()

    proj_run_dir = RUNS_DIR / project
    proj_run_dir.mkdir(parents=True, exist_ok=True)
    log_file = proj_run_dir / f"run_{model.replace('/', '_')}_{datetime.now().strftime('%H%M')}.log"

    cmd = [
        sys.executable, str(PIPELINE),
        "--project", project,
        "--model", model,
        "--port", str(port),
    ]

    try:
        env = os.environ.copy()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,
            cwd=str(ROOT),
            env=env,
        )

        log_file.write_text(result.stdout + "\n" + result.stderr)

        # Find result.json in the project's run directory
        result_files = list(proj_run_dir.rglob("result.json"))
        if result_files:
            newest = max(result_files, key=lambda f: f.stat().st_mtime)
            elapsed = time.time() - start
            return project, True, str(newest), elapsed, None
        else:
            elapsed = time.time() - start
            return project, False, None, elapsed, "No result.json found"

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        return project, False, None, elapsed, "TIMEOUT (30min)"
    except Exception as e:
        elapsed = time.time() - start
        return project, False, None, elapsed, str(e)[:200]


def main():
    parser = argparse.ArgumentParser(description="Run UI2App pipeline on all projects")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--projects", nargs="*", default=None,
                        help="Project prefixes to run (e.g., 01 03 09). Default: all")
    parser.add_argument("--base-port", type=int, default=BASE_PORT,
                        help=f"Starting port for dev servers (default {BASE_PORT})")
    args = parser.parse_args()

    discovered = discover_projects()
    all_names = sorted(discovered.keys())

    if args.projects:
        projects = [p for p in all_names if any(p.startswith(prefix) for prefix in args.projects)]
    else:
        projects = all_names

    if not projects:
        print(f"No projects found in data/apps/ — run `python data/sync_dataset.py` first.")
        return

    print(f"{'='*70}")
    print(f"UI2App — Batch Run")
    print(f"Model:       {args.model}")
    print(f"Projects:    {len(projects)}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Started:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    results = []
    start_all = time.time()

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {}
        for i, project in enumerate(projects):
            port = args.base_port + i
            future = executor.submit(run_one, project, args.model, port)
            futures[future] = project
            print(f"  Submitted: {project} (port {port})")

        print(f"\nWaiting for {len(futures)} jobs...\n")

        for future in as_completed(futures):
            project, success, result_path, elapsed, error = future.result()
            status = "OK" if success else "FAIL"
            print(f"  [{status}] {project:40s} {elapsed:6.0f}s", end="")

            if success and result_path:
                try:
                    r = json.loads(Path(result_path).read_text())
                    exec_str = "PASS" if r.get("exec_at1") or r.get("exec_at3") else "FAIL"
                    vfs_score = r.get("vfs_at1", 0)
                    print(f"  Exec={exec_str} vfs={vfs_score}")
                except Exception as e:
                    print(f"  (result parse error: {e})")
            else:
                print(f"  {error}")

            results.append({
                "project": project,
                "success": success,
                "result_path": result_path,
                "elapsed": round(elapsed, 1),
                "error": error,
            })

    total_time = time.time() - start_all
    ok = sum(1 for r in results if r["success"])

    print(f"\n{'='*70}")
    print(f"Completed: {ok}/{len(projects)} succeeded in {total_time:.0f}s")
    print(f"{'='*70}")

    # Save batch summary
    summary = {
        "model": args.model,
        "timestamp": datetime.now().isoformat(),
        "total_time": round(total_time, 1),
        "success_count": ok,
        "total_count": len(projects),
        "results": results,
    }

    detail_rows = []
    for r in results:
        if r["success"] and r["result_path"]:
            try:
                d = json.loads(Path(r["result_path"]).read_text())
                detail_rows.append({
                    "project": r["project"],
                    "exec_at1": d.get("exec_at1"),
                    "exec_at3": d.get("exec_at3"),
                    "debug_rounds": d.get("debug_rounds", 0),
                    "coverage": d.get("coverage_at1") or d.get("coverage_at3", 0),
                    "vfs_matched": d.get("vfs_matched_at1") or d.get("vfs_matched_at3", 0),
                    "vfs_per_app": d.get("vfs_per_app_at1") or d.get("vfs_per_app_at3", 0),
                    "vfs": d.get("vfs_at1", 0),
                })
            except Exception as e:
                print(f"  [warn] failed to parse {r['result_path']}: {e}")
    summary["details"] = detail_rows

    summary_path = SCRIPT_DIR / f"batch_summary_{args.model.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Summary: {summary_path}")

    if detail_rows:
        print(f"\n{'Project':40s} {'Exec':5s} {'Cov':5s} {'VFSm':5s} {'VFSp':5s} {'VFS':5s}")
        print("-" * 70)
        for d in sorted(detail_rows, key=lambda x: x["project"]):
            exec_str = "@1" if d["exec_at1"] else ("@3" if d["exec_at3"] else "FAIL")
            print(f"{d['project']:40s} {exec_str:5s} {d['coverage']:5.1f} {d['vfs_matched']:5.1f} "
                  f"{d['vfs_per_app']:5.1f} {d['vfs']:5.1f}")


if __name__ == "__main__":
    main()
