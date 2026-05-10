#!/usr/bin/env python3
"""
UI2App Pipeline — unified entry point.

Orchestrates code generation (Stage 1-3) and evaluation (Stage 4-6):
  Stage 1: Plan → Gen (multi-turn, screenshots sent once)
  Stage 2: Exec check (vite build + Playwright render)
  Stage 3: Self-debug (same model, max 3 rounds)
  Stage 4: Coverage (route matching) (structural + render)
  Stage 5: VFS_matched = (Size + Text + Position + Color) / 4 × 100  (DOM-to-DOM)
  Stage 6: VFS = Exec × VFS_per_app

Results are saved alongside input in benchmark/projects/phase*/{project}/{model}/run_*/

Usage:
  python benchmark/pipeline.py --project 01_fashion-ecommerce --model gpt-4o
  python benchmark/pipeline.py --project 01_fashion-ecommerce --model gpt-4o --skip-gen --run-dir path/to/run
"""

import argparse, json, os, sys
from pathlib import Path
from datetime import datetime

# Ensure benchmark/ is on sys.path so support/ and stages/ resolve when invoked directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from support.projects import ROOT, PROJECTS, ENTRY_FILES, RUNS_DIR, SOURCES_DIR
from support.llm import get_client, log_attempt, CURRENT_PROJECT, CURRENT_MODEL
from support.server import start_server, stop_server
from support.auth import detect_gates
from stages.generate import run_generation
from stages.coverage import compute_coverage
from stages.visual import compute_vfs_dom

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ─── Stage 6: Visual Fidelity Score ──────────────────────────────────────────

def compute_vfs(exec_pass, vfs_per_app):
    """VFS = Executability × VFS_per_app (as percentage)."""
    return round(exec_pass * (vfs_per_app / 100) * 100, 1)


def compute_vfs_per_app(vfs_matched, n_matched, n_total):
    """VFS_per_app = VFS_matched × (n_matched / n_total), penalizing unmatched pages."""
    if n_total == 0:
        return 0.0
    return round(vfs_matched * n_matched / n_total, 1)


# ─── Stage 4-6 evaluation orchestrator ───────────────────────────────────────

def run_evaluation(ss_dir, gen_dir, source_dir, manifest, exec_at1, exec_at3, gen_port=5280):
    """Run Stage 4-6. Returns dict with all evaluation metrics.

    Args:
        ss_dir: screenshot input directory
        gen_dir: generated code directory
        source_dir: original project source directory
        manifest: parsed manifest.json
        exec_at1, exec_at3: Exec results from generation stage
        gen_port: base port for dev servers
    """
    coverage_at1 = 0; vfs_matched_at1 = 0; coverage_at3 = 0; vfs_matched_at3 = 0
    coverage_details = {}; matched_pairs = []; vfs_details = []
    unmatched_gen_routes = []; empty_gen_routes = []
    gate_diag = {"has_gate": False, "pattern": None, "gate_component": None, "redirects": []}

    port_orig = gen_port + 10
    port_gen_vfs = gen_port + 11
    source_port_cov = gen_port + 12  # distinct from VFS ports

    if exec_at1:
        print(f"\n--- Stage 4: Coverage (@1, DOM-align) ---")
        (coverage_at1, matched_pairs, coverage_details, _rendered,
         unmatched_gen_routes, empty_gen_routes) = compute_coverage(
            ss_dir, gen_dir, gen_port, source_dir=source_dir, source_port=source_port_cov)
        gate_diag = _rendered.get("__gate_info__", gate_diag) if isinstance(_rendered, dict) else gate_diag
        blocks_cache = _rendered.get("__blocks_cache__") if isinstance(_rendered, dict) else None
        print(f"\n--- Stage 5: VFS DOM-to-DOM (@1) ---")
        gate_vfs = detect_gates(gen_dir) if gate_diag.get("has_gate") else None
        vfs_matched_at1, vfs_details = compute_vfs_dom(matched_pairs, source_dir, gen_dir, port_orig, port_gen_vfs,
                                            gate_info=gate_vfs, blocks_cache=blocks_cache)
        coverage_at3 = coverage_at1; vfs_matched_at3 = vfs_matched_at1
    elif exec_at3:
        print(f"\n--- Stage 4: Coverage (@3, DOM-align) ---")
        (coverage_at3, matched_pairs, coverage_details, _rendered,
         unmatched_gen_routes, empty_gen_routes) = compute_coverage(
            ss_dir, gen_dir, gen_port, source_dir=source_dir, source_port=source_port_cov)
        gate_diag = _rendered.get("__gate_info__", gate_diag) if isinstance(_rendered, dict) else gate_diag
        blocks_cache = _rendered.get("__blocks_cache__") if isinstance(_rendered, dict) else None
        print(f"\n--- Stage 5: VFS DOM-to-DOM (@3) ---")
        gate_vfs = detect_gates(gen_dir) if gate_diag.get("has_gate") else None
        vfs_matched_at3, vfs_details = compute_vfs_dom(matched_pairs, source_dir, gen_dir, port_orig, port_gen_vfs,
                                            gate_info=gate_vfs, blocks_cache=blocks_cache)
    else:
        print(f"\n--- Stages 4-5: SKIPPED (Exec failed) ---")

    # ── Stage 6: Visual Fidelity Score ──
    n_total = len(manifest["screenshots"])
    if exec_at1:
        n_matched = len([d for d in coverage_details.values() if d.get("matched")])
        vfs_per_app_at1 = compute_vfs_per_app(vfs_matched_at1, n_matched, n_total)
        vfs_per_app_at3 = vfs_per_app_at1
        score_at1 = compute_vfs(1, vfs_per_app_at1)
        score_at3 = score_at1
    elif exec_at3:
        n_matched = len([d for d in coverage_details.values() if d.get("matched")])
        vfs_per_app_at3 = compute_vfs_per_app(vfs_matched_at3, n_matched, n_total)
        vfs_per_app_at1 = 0.0
        score_at1 = 0.0
        score_at3 = compute_vfs(1, vfs_per_app_at3)
    else:
        vfs_per_app_at1 = 0.0; vfs_per_app_at3 = 0.0
        score_at1 = 0.0; score_at3 = 0.0

    return {
        "coverage_at1": coverage_at1, "coverage_at3": coverage_at3,
        "vfs_matched_at1": vfs_matched_at1, "vfs_matched_at3": vfs_matched_at3,
        "vfs_per_app_at1": vfs_per_app_at1, "vfs_per_app_at3": vfs_per_app_at3,
        "vfs_at1": score_at1, "vfs_at3": score_at3,
        "coverage_details": coverage_details, "vfs_details": vfs_details,
        "matched_pairs": matched_pairs,
        "unmatched_gen_routes": unmatched_gen_routes,
        "empty_gen_routes": empty_gen_routes,
        "gate_info": gate_diag,
    }


# ─── Top-level pipeline (Stage 1-6) ──────────────────────────────────────────


def run_pipeline(project_name, model_name, gen_port=5280, skip_gen=False, existing_run_dir=None, skip_eval=False):
    # Bind context for cost_log + attempts log
    CURRENT_PROJECT.set(project_name)
    CURRENT_MODEL.set(model_name)
    mode = "skip-gen" if skip_gen else ("skip-eval" if skip_eval else "full")
    log_attempt("started", project=project_name, model=model_name,
                mode=mode, port=gen_port)

    project = PROJECTS[project_name]
    ss_dir = project["screenshot_dir"]   # data/apps/<app_id>/
    try:
        manifest = json.loads((ss_dir / "manifest.json").read_text())
    except json.JSONDecodeError as e:
        log_attempt("failed", error=f"manifest.json parse error: {e}", mode=mode)
        print(f"ERROR: manifest.json invalid: {e}")
        sys.exit(1)

    # Source repo for VFS Stage 5 lives at data/sources/<app_id>/. If absent,
    # pipeline gracefully degrades to cascade Coverage + OCR VFS fallback.
    convention_source = SOURCES_DIR / project_name
    if convention_source.is_dir():
        source_dir = convention_source
    else:
        print(f"  [Note] source_dir not found at {convention_source}")
        print(f"         Stage 4 → cascade fallback; Stage 5 → OCR fallback on input PNGs.")
        print(f"         For full DOM-based VFS, run: python data/download_sources.py")
        source_dir = None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_short = model_name

    if existing_run_dir:
        run_dir = Path(existing_run_dir)
        gen_dir = run_dir / "generated"
        if not gen_dir.exists():
            print(f"ERROR: {gen_dir} does not exist")
            sys.exit(1)
    else:
        # data/runs/<app_id>/<model>/run_<ts>/
        run_dir = RUNS_DIR / project_name / model_short / f"run_{timestamp}"
        gen_dir = run_dir / "generated"

    print(f"\n{'='*70}")
    print(f"UI2App Evaluation Pipeline")
    print(f"Project: {project_name}")
    print(f"Model:   {model_name}")
    print(f"Source:  {source_dir}")
    print(f"Output:  {run_dir}")
    print(f"{'='*70}")

    client = get_client()

    # ── Stage 1-3: Generation ──
    # --skip-gen means "don't regenerate code" AND "don't re-verify Exec" —
    # reuse Exec values from prior result.json. Avoids re-running vite build +
    # self-debug (wasted LLM $, and risks overwriting result.json with Exec=False
    # when node_modules was cleaned up between runs).
    if skip_gen:
        prior_path = run_dir / "result.json"
        if not prior_path.exists():
            print(f"ERROR: --skip-gen requires prior result.json at {prior_path}")
            print(f"       (run with --skip-eval first to generate code, or drop --skip-gen)")
            log_attempt("failed", error="skip-gen without prior result.json", mode=mode)
            sys.exit(1)
        prior = json.loads(prior_path.read_text())
        exec_at1        = prior["exec_at1"]
        exec_at3        = prior["exec_at3"]
        debug_rounds    = prior.get("debug_rounds", 0)
        debug_log       = prior.get("debug_log", [])
        # Schema-rename: prior runs may still have the legacy "l1_fail_info" key.
        exec_fail_info  = prior.get("exec_fail_info") or prior.get("l1_fail_info")
        print(f"\n--- Stages 1-3: SKIPPED (reusing Exec from prior result.json) ---")
        print(f"  Exec@1={exec_at1}  Exec@3={exec_at3}  debug_rounds={debug_rounds}")
    else:
        exec_at1, exec_at3, debug_rounds, debug_log, exec_fail_info = run_generation(
            client, model_name, ss_dir, run_dir, gen_dir,
            port=gen_port, skip_gen=False,
        )

    # ── Stage 4-6: Evaluation ──
    if skip_eval:
        print("\n--- Stages 4-6 SKIPPED (--skip-eval) ---")
        eval_result = {
            "coverage_at1": None, "coverage_at3": None,
            "vfs_matched_at1": None, "vfs_matched_at3": None,
            "vfs_per_app_at1": None, "vfs_per_app_at3": None,
            "vfs_at1": None, "vfs_at3": None,
            "coverage_details": None, "vfs_details": None,
            "matched_pairs": [],
            "eval_skipped": True,
        }
    else:
        eval_result = run_evaluation(
            ss_dir, gen_dir, source_dir, manifest,
            exec_at1, exec_at3, gen_port=gen_port,
        )

    # ── Print Summary ──
    print(f"\n{'='*70}")
    print(f"Results: {project_name} x {model_name}")
    print(f"{'='*70}")
    print(f"  Exec@1={exec_at1}  Exec@3={exec_at3}  DebugRounds={debug_rounds}")
    if skip_eval:
        print(f"  (Eval skipped — run pipeline.py --skip-gen --run-dir {run_dir} to score later)")
    else:
        print(f"  Coverage={eval_result['coverage_at1'] or eval_result['coverage_at3']} (diagnostic)")
        print(f"  VFS_matched={eval_result['vfs_matched_at1'] or eval_result['vfs_matched_at3']}  "
              f"VFS_per_app={eval_result['vfs_per_app_at1'] or eval_result['vfs_per_app_at3']}")
        print(f"  Visual Fidelity Score@1={eval_result['vfs_at1']}  Visual Fidelity Score@3={eval_result['vfs_at3']}")
    print(f"{'='*70}")

    # ── Save Result ──
    matched_pairs = eval_result.pop("matched_pairs")

    result = {
        "project": project_name,
        "model": model_name,
        "pipeline_version": "1.0",
        "vfs_method": "DOM-to-DOM",
        "timestamp": timestamp,
        "exec_at1": exec_at1, "exec_at3": exec_at3,
        "exec_fail_info": exec_fail_info,
        "debug_rounds": debug_rounds, "debug_log": debug_log,
        **eval_result,
    }
    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    # Create paired symlinks for visual comparison.
    # Always refresh (delete stale symlinks) so re-running eval produces
    # accurate pairings — previous versions used try/except FileExistsError
    # which silently kept outdated links from an earlier (possibly buggy) run.
    if not skip_eval:
        paired_dir = run_dir / "paired"
        paired_dir.mkdir(exist_ok=True)
        # Remove any stale *_input.png / *_gen.png entries from prior eval runs
        for stale in paired_dir.iterdir():
            if stale.is_symlink() and stale.name.endswith(("_input.png", "_gen.png")):
                stale.unlink()
        for mp in matched_pairs:
            inp_path = Path(mp["input"])
            gen_path = Path(mp["generated"])
            base = mp["input_name"].replace(".png", "")
            if inp_path.exists() and gen_path.exists():
                os.symlink(inp_path.resolve(), paired_dir / f"{base}_input.png")
                os.symlink(gen_path.resolve(), paired_dir / f"{base}_gen.png")

    if not existing_run_dir:
        latest_link = run_dir.parent / "latest"
        if latest_link.is_symlink():
            latest_link.unlink()
        try:
            os.symlink(run_dir.name, latest_link)
        except Exception as e:
            print(f"  [warn] latest symlink skipped: {e}")

    print(f"\nSaved: {result_path}")
    log_attempt("ok", run_dir=run_dir, mode=mode,
                exec_at1=exec_at1, exec_at3=exec_at3,
                ui2app=eval_result.get('vfs_at3'))
    return result


# ─── CLI ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UI2App Evaluation Pipeline")
    parser.add_argument("--project", required=True, choices=list(PROJECTS.keys()))
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--port", type=int, default=5280)
    parser.add_argument("--skip-gen", action="store_true")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Run only code generation; skip Coverage/VFS evaluation. "
                             "Use --skip-gen --run-dir later to score.")
    parser.add_argument("--run-dir", type=str, default=None)
    args = parser.parse_args()

    from support.llm import log_attempt as _la
    try:
        run_pipeline(args.project, args.model, args.port, args.skip_gen, args.run_dir,
                     skip_eval=args.skip_eval)
    except SystemExit:
        raise
    except BaseException as _e:
        _la("failed", project=args.project, model=args.model,
            error=f"{type(_e).__name__}: {_e}",
            mode=("skip-gen" if args.skip_gen else "skip-eval" if args.skip_eval else "full"))
        raise
