"""
UI2App DOM block extraction + OCR fallback + render probe + Stage 5 (VFS DOM-to-DOM).
"""

import json, os, re, subprocess, sys, hashlib, time
from pathlib import Path

import numpy as np
from support.server import start_server, stop_server
from support.projects import ENTRY_FILES
from support.blocks import extract_blocks, _EXTRACT_BLOCKS_JS, compute_vfs_page
from support.auth import detect_gates, _try_dismiss_gate



def _is_rendered(probe):
    """Multi-signal render-validity check (relaxed v2).

    Earlier versions used `all >= 5` as a hard structural prereq, which
    incorrectly rejected legitimate splash/intro/CTA pages with deep text but
    shallow DOM (e.g., a `<div><button>Skip</button><p>...</p></div>` Intro
    component has only `all=3` despite 90+ chars of meaningful text). The
    revised rule keeps a tiny `all >= 3` floor (rejects truly empty pages)
    and otherwise accepts any single sufficient signal:
      (a) >= 20 chars of plain text                     [text-rich / splash]
      (b) >= 2 visual primitives (svg/img/canvas/video) [icon/canvas pages]
      (c) >= 2 interactive primitives (button/input/...) [form/CTA pages]
    """
    if probe["all"] < 3:
        return False
    return (probe["text"] >= 20 or probe["visuals"] >= 2 or probe["interactive"] >= 2)



# ─── Stage 5: VFS DOM-to-DOM ───────────────────────────────────────

import filelock as _filelock_mod

def _gated_extract_blocks(page, url, gate_info, wait_ms=3000, timeout=15000,
                          route_path="/"):
    """extract_blocks with asymmetric gate dismissal.

    Policy mirrors `compute_coverage`: dismiss the gate on every route *except* the
    home `/`, where the reference screenshot itself was captured with the gate
    showing (the typical Phineas-style `shouldPlayIntro = pathname === '/'`
    pattern). Comparing gate-vs-gate at home and content-vs-content elsewhere
    keeps the gen and ref sides symmetric per route.
    """
    from support.blocks import _EXTRACT_BLOCKS_JS
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=timeout)
        page.wait_for_timeout(min(wait_ms, 1500))
        is_home = route_path.rstrip("/") in ("", "/")
        if gate_info and gate_info.get("has_gate") and not is_home:
            _try_dismiss_gate(page, gate_info["dismiss_selectors"])
        page.wait_for_timeout(wait_ms)
        return page.evaluate(_EXTRACT_BLOCKS_JS) or []
    except Exception as e:
        print(f"    [BlockExtract] ERROR {url}: {str(e)[:60]}")
        return []


def compute_vfs_dom(matched_pairs, source_dir, gen_dir, port_orig=5290, port_gen=5291,
                   gate_info=None, blocks_cache=None):
    """Stage 5 VFS: DOM-to-DOM 4-metric evaluation.

    `blocks_cache`: optional dict `{"ref_blocks": {input_filename: [..]}, "gen_blocks": {route_def: [..]}}`
    populated by dom_align in compute_coverage. When present, VFS skips the dev-server starts
    and uses cached blocks directly — saves ~2-3 minutes per project.
    """
    if not matched_pairs:
        return 0.0, []

    # Reviewer fallback: source repo not available → OCR input PNGs for ref blocks,
    # gen side still rendered via vite + Playwright. Produces real (non-zero) VFS,
    # less precise than DOM-based reference because OCR loses fg_color
    # (Color metric placeholder) and merges adjacent text by line, not by DOM tag.
    if source_dir is None:
        from support.blocks import _ocr_input_blocks
        print(f"  [VFS-DOM] No source_dir → OCR-on-input-PNG fallback for ref blocks")
        srv_gen, url_gen = start_server(gen_dir, port_gen)
        if url_gen is None:
            print(f"  [VFS-DOM] Gen-side server failed to start; skipping VFS")
            stop_server(srv_gen)
            return 0.0, []
        scores, details = [], []
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(viewport={"width": 1440, "height": 900})
                for pair in matched_pairs:
                    gen_url = pair.get("gen_url", pair.get("gen_route", "/"))
                    url_g = f"{url_gen}{gen_url}"
                    blocks_orig = _ocr_input_blocks(pair["input"])
                    blocks_gen = _gated_extract_blocks(page, url_g, gate_info, wait_ms=3000,
                                                       route_path=pair.get("gen_route", gen_url))
                    page_vfs, metrics = compute_vfs_page(blocks_orig, blocks_gen)
                    scores.append(page_vfs)
                    details.append({"pair": pair["input_name"], "vfs": page_vfs, "ref_source": "ocr", **metrics})
                    print(f"    {pair['input_name']:35s} S={metrics['size']:.2f} T={metrics['text']:.2f} "
                          f"P={metrics['position']:.2f} C={metrics['color']:.2f} → {page_vfs}  [ocr]")
                browser.close()
        finally:
            stop_server(srv_gen)
        vfs = round(sum(scores) / max(len(scores), 1), 1) if scores else 0.0
        print(f"  [VFS-DOM] {vfs} (from {len(scores)} matched pairs, OCR fallback)")
        return vfs, details

    # Fast path: blocks already extracted by dom_align in compute_coverage.
    if blocks_cache and blocks_cache.get("ref_blocks") and blocks_cache.get("gen_blocks"):
        ref_b = blocks_cache["ref_blocks"]
        gen_b = blocks_cache["gen_blocks"]
        scores, details = [], []
        for pair in matched_pairs:
            r = ref_b.get(pair["input_name"], [])
            g = gen_b.get(pair["gen_route"], [])
            page_vfs, metrics = compute_vfs_page(r, g)
            scores.append(page_vfs)
            details.append({"pair": pair["input_name"], "vfs": page_vfs, **metrics})
            print(f"    {pair['input_name']:35s} S={metrics['size']:.2f} T={metrics['text']:.2f} "
                  f"P={metrics['position']:.2f} C={metrics['color']:.2f} → {page_vfs}")
        vfs = round(sum(scores) / max(len(scores), 1), 1) if scores else 0.0
        print(f"  [VFS-DOM] {vfs} (from {len(scores)} matched pairs, blocks_cache hit)")
        return vfs, details

    source_dir = Path(source_dir)
    if (source_dir / "package.json").exists() and not (source_dir / "node_modules").exists():
        print(f"  [VFS-DOM] Installing deps for original project...")
        subprocess.run(["pnpm", "install"], cwd=str(source_dir), capture_output=True, timeout=600)

    # Lock source directory to prevent concurrent server starts on the same source project
    lock_path = source_dir / ".ui2app_vfs.lock"
    lock = _filelock_mod.FileLock(lock_path, timeout=300)
    lock.acquire()
    print(f"  [VFS-DOM] Acquired lock on {source_dir.name}")

    srv_orig, url_orig = start_server(source_dir, port_orig)
    srv_gen, url_gen = start_server(gen_dir, port_gen)

    if url_orig is None or url_gen is None:
        print(f"  [VFS-DOM] Server failed to start (orig={url_orig is not None}, gen={url_gen is not None})")
        stop_server(srv_orig)
        stop_server(srv_gen)
        lock.release()
        return 0.0, []

    scores = []
    details = []
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 900})

            for pair in matched_pairs:
                input_route = pair.get("input_route", "/")
                gen_url = pair.get("gen_url", pair.get("gen_route", "/"))

                url_o = f"{url_orig}{input_route.split('?')[0]}"
                url_g = f"{url_gen}{gen_url}"

                # Reference side never has a gate (the source app builds and ships
                # to production); only the gen side may need dismissal.
                blocks_orig = extract_blocks(page, url_o, wait_ms=4000, timeout=30000)
                blocks_gen = _gated_extract_blocks(page, url_g, gate_info, wait_ms=3000,
                                                   route_path=pair.get("gen_route", gen_url))

                page_vfs, metrics = compute_vfs_page(blocks_orig, blocks_gen)

                scores.append(page_vfs)
                detail = {"pair": pair["input_name"], "vfs": page_vfs, **metrics}
                details.append(detail)
                print(f"    {pair['input_name']:35s} S={metrics['size']:.2f} T={metrics['text']:.2f} "
                      f"P={metrics['position']:.2f} C={metrics['color']:.2f} → {page_vfs}")

            browser.close()
    finally:
        stop_server(srv_orig)
        stop_server(srv_gen)
        lock.release()

    vfs = round(sum(scores) / max(len(scores), 1), 1) if scores else 0.0
    print(f"  [VFS-DOM] {vfs} (from {len(scores)} matched pairs)")
    return vfs, details

