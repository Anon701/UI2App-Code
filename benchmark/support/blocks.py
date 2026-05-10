#!/usr/bin/env python3
"""
VFS DOM-to-DOM visual evaluation for UI2App.

4 objective metrics: Size + Text + Position + Color
VFS_matched = (Size + Text + Position + Color) / 4 × 100

Provides two usage modes:
  1. Library: evaluate.py calls extract_blocks, match_blocks, compute_vfs_page()
  2. CLI: standalone batch evaluation over existing experiment results

Optional: compute_clip() available as supplementary metric (not in default VFS).

Usage (CLI):
  python -m benchmark.support.blocks --experiments-dir experiments/exp3_vlm_evaluation/results
  python -m benchmark.support.blocks --experiments-dir ... --projects 01 04
"""

import argparse, json, os, re, signal, subprocess, sys, time
from pathlib import Path
from difflib import SequenceMatcher
from datetime import datetime

import numpy as np
from scipy.optimize import linear_sum_assignment


# ─── DOM Block Extraction ──────────────────────────────────────────

_EXTRACT_BLOCKS_JS = """
() => {
    const blocks = [];
    const vw = window.innerWidth || 1440;
    const vh = window.innerHeight || 900;

    const walker = document.createTreeWalker(
        document.body,
        NodeFilter.SHOW_TEXT,
        {
            acceptNode: (node) => {
                const text = node.textContent.trim();
                if (!text || text.length > 500) return NodeFilter.FILTER_REJECT;
                const el = node.parentElement;
                if (!el) return NodeFilter.FILTER_REJECT;
                const style = window.getComputedStyle(el);
                if (style.display === 'none' || style.visibility === 'hidden' ||
                    parseFloat(style.opacity) === 0)
                    return NodeFilter.FILTER_REJECT;
                return NodeFilter.FILTER_ACCEPT;
            }
        }
    );

    while (walker.nextNode()) {
        const node = walker.currentNode;
        const el = node.parentElement;
        const rect = el.getBoundingClientRect();
        if (rect.width < 2 || rect.height < 2) continue;
        if (rect.bottom < 0 || rect.top > vh * 3) continue;

        const style = window.getComputedStyle(el);
        function parseColor(c) {
            const m = c.match(/\\d+/g);
            return m ? m.slice(0, 3).map(Number) : [0, 0, 0];
        }
        blocks.push({
            text: node.textContent.trim().toLowerCase().substring(0, 200),
            bbox: { x: rect.left / vw, y: rect.top / vh, w: rect.width / vw, h: rect.height / vh },
            fg_color: parseColor(style.color),
            tag: el.tagName.toLowerCase(),
        });
    }

    // Merge adjacent same-row same-tag text blocks
    const merged = [];
    for (let i = 0; i < blocks.length; i++) {
        const b = blocks[i];
        if (merged.length > 0) {
            const prev = merged[merged.length - 1];
            const sameRow = Math.abs(prev.bbox.y - b.bbox.y) < 0.02;
            const adjacent = (b.bbox.x - (prev.bbox.x + prev.bbox.w)) < 0.02;
            if (sameRow && adjacent && prev.tag === b.tag) {
                prev.text += ' ' + b.text;
                prev.bbox.w = (b.bbox.x + b.bbox.w) - prev.bbox.x;
                continue;
            }
        }
        merged.push({...b});
    }
    return merged;
}
"""


def extract_blocks(pw_page, url, wait_ms=3000, timeout=15000):
    """Navigate to URL and extract DOM text blocks via Playwright."""
    try:
        pw_page.goto(url, wait_until="domcontentloaded", timeout=timeout)
        pw_page.wait_for_timeout(wait_ms)
        return pw_page.evaluate(_EXTRACT_BLOCKS_JS) or []
    except Exception as e:
        print(f"    [BlockExtract] ERROR {url}: {str(e)[:60]}")
        return []


# ─── Block Matching (Hungarian Algorithm) ─────────────────────────

def match_blocks(blocks_orig, blocks_gen, min_text_sim=0.3):
    """Match blocks using Hungarian algorithm on text similarity cost matrix."""
    n, m = len(blocks_orig), len(blocks_gen)
    if n == 0 or m == 0:
        return []

    cost = np.ones((n, m))
    for i, bo in enumerate(blocks_orig):
        for j, bg in enumerate(blocks_gen):
            sim = SequenceMatcher(None, bo["text"], bg["text"]).ratio()
            cost[i][j] = 1.0 - sim

    size = max(n, m)
    padded = np.ones((size, size))
    padded[:n, :m] = cost

    row_ind, col_ind = linear_sum_assignment(padded)

    pairs = []
    for i, j in zip(row_ind, col_ind):
        if i >= n or j >= m:
            continue
        text_sim = 1.0 - cost[i][j]
        if text_sim >= min_text_sim:
            pairs.append({
                "orig": blocks_orig[i],
                "gen": blocks_gen[j],
                "text_sim": text_sim,
            })

    return pairs


# ─── 4 Core Metrics ──────────────────────────────────────────────

def compute_size(pairs, blocks_orig, blocks_gen):
    """Size Score: matched area coverage."""
    def area(b):
        return b["bbox"]["w"] * b["bbox"]["h"]

    if not pairs:
        return 0.0

    matched_area = sum((area(p["orig"]) + area(p["gen"])) / 2 for p in pairs)
    total_orig = sum(area(b) for b in blocks_orig)
    total_gen = sum(area(b) for b in blocks_gen)
    total = (total_orig + total_gen) / 2

    if total == 0:
        return 0.0
    return min(1.0, matched_area / total)


def compute_text(pairs):
    """Text Score: mean text similarity of matched pairs."""
    if not pairs:
        return 0.0
    return sum(p["text_sim"] for p in pairs) / len(pairs)


def compute_position(pairs):
    """Position Score: 1 - mean Chebyshev distance of matched block centers."""
    if not pairs:
        return 0.0

    scores = []
    for p in pairs:
        bo, bg = p["orig"]["bbox"], p["gen"]["bbox"]
        cx_o = bo["x"] + bo["w"] / 2
        cy_o = bo["y"] + bo["h"] / 2
        cx_g = bg["x"] + bg["w"] / 2
        cy_g = bg["y"] + bg["h"] / 2
        dist = max(abs(cx_o - cx_g), abs(cy_o - cy_g))
        scores.append(max(0.0, 1.0 - dist))

    return sum(scores) / len(scores)


def compute_color(pairs):
    """Color Score: mean CIEDE2000 color similarity of matched block foreground colors."""
    if not pairs:
        return 0.0

    try:
        from skimage.color import rgb2lab, deltaE_ciede2000
    except ImportError:
        # Fallback: simple Euclidean RGB distance
        scores = []
        for p in pairs:
            fg_o = np.array(p["orig"]["fg_color"], dtype=float)
            fg_g = np.array(p["gen"]["fg_color"], dtype=float)
            dist = np.linalg.norm(fg_o - fg_g) / 441.67  # max RGB distance = sqrt(3*255^2)
            scores.append(max(0.0, 1.0 - dist))
        return sum(scores) / len(scores)

    scores = []
    for p in pairs:
        fg_o = np.array(p["orig"]["fg_color"], dtype=float).reshape(1, 1, 3) / 255.0
        fg_g = np.array(p["gen"]["fg_color"], dtype=float).reshape(1, 1, 3) / 255.0
        lab_o = rgb2lab(fg_o)
        lab_g = rgb2lab(fg_g)
        delta_e = deltaE_ciede2000(lab_o, lab_g)[0][0]
        scores.append(max(0.0, 1.0 - delta_e / 100.0))

    return sum(scores) / len(scores)


# ─── Page-Level VFS Score ──────────────────────────────────────────

def compute_vfs_page(blocks_orig, blocks_gen):
    """Compute VFS_matched for a single page from extracted DOM blocks.

    Returns (page_vfs_score, detail_dict).
    VFS_matched = (Size + Text + Position + Color) / 4 × 100
    """
    pairs = match_blocks(blocks_orig, blocks_gen)

    s = compute_size(pairs, blocks_orig, blocks_gen)
    t = compute_text(pairs)
    pos = compute_position(pairs)
    c = compute_color(pairs)
    page_vfs = round((s + t + pos + c) / 4 * 100, 1)

    return page_vfs, {
        "size": round(s, 3),
        "text": round(t, 3),
        "position": round(pos, 3),
        "color": round(c, 3),
        "n_blocks_orig": len(blocks_orig),
        "n_blocks_gen": len(blocks_gen),
        "n_matched": len(pairs),
    }


# ─── Optional: CLIP Score (supplementary, not in default VFS) ─────

def compute_clip(screenshot_orig, screenshot_gen):
    """CLIP Score: ViT-B/32 cosine similarity between full-page screenshots.

    NOT included in the default VFS formula. Use as supplementary metric.
    """
    try:
        import torch
        import open_clip
        from PIL import Image

        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        model.eval()

        img_o = preprocess(Image.open(screenshot_orig)).unsqueeze(0)
        img_g = preprocess(Image.open(screenshot_gen)).unsqueeze(0)

        with torch.no_grad():
            feat_o = model.encode_image(img_o)
            feat_g = model.encode_image(img_g)
            feat_o = feat_o / feat_o.norm(dim=-1, keepdim=True)
            feat_g = feat_g / feat_g.norm(dim=-1, keepdim=True)
            sim = (feat_o @ feat_g.T).item()

        return max(0.0, sim)
    except ImportError:
        print("    [CLIP] open_clip/torch not installed, skipping")
        return None


# ─── Server Management (for CLI mode) ────────────────────────────

def _start_server(proj_dir, port, cmd=None):
    """Start dev server, return (process, base_url)."""
    if cmd is None:
        pkg_path = proj_dir / "package.json"
        if pkg_path.exists():
            pkg = json.loads(pkg_path.read_text())
            deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
            scripts = pkg.get("scripts", {})
            if "next" in deps:
                cmd = ["npx", "next", "dev", "-p", str(port)]
            elif "astro" in deps:
                cmd = ["npx", "astro", "dev", "--port", str(port)]
            elif "react-scripts" in deps:
                cmd = ["npx", "react-scripts", "start"]
                os.environ["PORT"] = str(port)
                os.environ["BROWSER"] = "none"
            else:
                cmd = ["npx", "vite", "--port", str(port)]
        else:
            cmd = ["npx", "vite", "--port", str(port)]

    srv = subprocess.Popen(
        cmd, cwd=str(proj_dir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    import urllib.request
    url = f"http://localhost:{port}"
    for _ in range(25):
        time.sleep(1)
        try:
            urllib.request.urlopen(url, timeout=3)
            return srv, url
        except:
            pass
    return srv, url


def _stop_server(srv):
    if srv:
        try:
            os.killpg(os.getpgid(srv.pid), signal.SIGTERM)
        except:
            try:
                srv.kill()
            except:
                pass


# ─── CLI: Batch Evaluation ────────────────────────────────────────

def eval_project_cli(exp_dir, port_orig=5300, port_gen=5301):
    """Evaluate one project (CLI mode). Returns (vfs_per_app, page_details)."""
    exp_dir = Path(exp_dir)
    input_dir = exp_dir / "input"
    manifest_path = input_dir / "manifest.json"

    if not manifest_path.exists():
        print(f"  SKIP {exp_dir.name}: no manifest")
        return None, None

    manifest = json.loads(manifest_path.read_text())
    screenshots = manifest.get("screenshots", [])
    source_dir = Path(manifest.get("source_dir", ""))

    run_dirs = sorted(exp_dir.rglob("generated"), key=lambda f: f.stat().st_mtime)
    if not run_dirs:
        print(f"  SKIP {exp_dir.name}: no generated code")
        return None, None
    gen_dir = run_dirs[-1]
    run_dir = gen_dir.parent

    result_path = run_dir / "result.json"
    coverage_details = {}
    if result_path.exists():
        old_result = json.loads(result_path.read_text())
        coverage_details = old_result.get("coverage_details", {})

    print(f"\n  Evaluating: {exp_dir.name}")
    print(f"    Source: {source_dir}")
    print(f"    Generated: {gen_dir}")
    print(f"    Pages: {len(screenshots)}")

    srv_orig, url_orig = _start_server(source_dir, port_orig)
    srv_gen, url_gen = _start_server(gen_dir, port_gen)

    page_scores = []
    page_details = []

    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 900})

            for ss in screenshots:
                fname = ss["filename"]
                route = ss.get("route", "/" + ss.get("label", ""))

                cov_info = coverage_details.get(fname, {})
                if cov_info.get("matched"):
                    gen_route = cov_info.get("gen_route", route)
                    gen_url = re.sub(r':(\w+)', '1', gen_route) if ":" in gen_route else gen_route
                else:
                    gen_url = route.split("?")[0]

                url_o = f"{url_orig}{route.split('?')[0]}"
                url_g = f"{url_gen}{gen_url}"

                try:
                    blocks_orig = extract_blocks(page, url_o, wait_ms=4000)
                    blocks_gen = extract_blocks(page, url_g, wait_ms=3000)
                    score, details = compute_vfs_page(blocks_orig, blocks_gen)
                    page_scores.append(score)
                    page_details.append({"page": fname, "route": route, "score": score, **details})
                    print(f"    {fname:40s} S={details['size']:.2f} T={details['text']:.2f} "
                          f"P={details['position']:.2f} C={details['color']:.2f} → {score}")
                except Exception as e:
                    page_scores.append(0.0)
                    page_details.append({"page": fname, "route": route, "score": 0.0, "error": str(e)[:100]})
                    print(f"    {fname:40s} ERROR: {str(e)[:60]}")

            browser.close()
    finally:
        _stop_server(srv_orig)
        _stop_server(srv_gen)

    n_total = len(screenshots)
    vfs_per_app = round(sum(page_scores) / max(n_total, 1), 1)
    print(f"    VFS_per_app = {vfs_per_app} ({len([s for s in page_scores if s > 0])}/{n_total} pages scored)")

    return vfs_per_app, page_details


def main():
    parser = argparse.ArgumentParser(description="VFS DOM-to-DOM visual evaluation")
    parser.add_argument("--experiments-dir", required=True, help="Path to experiment results directory")
    parser.add_argument("--projects", nargs="*", default=None, help="Project prefixes (e.g., 01 04)")
    args = parser.parse_args()

    exp_base = Path(args.experiments_dir)
    project_dirs = sorted(exp_base.glob("[0-9]*"))

    if args.projects:
        project_dirs = [d for d in project_dirs
                        if any(d.name.startswith(p) for p in args.projects)]

    print("=" * 70)
    print("VFS: DOM-to-DOM Visual Evaluation (4-metric)")
    print(f"Projects: {len(project_dirs)}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    all_results = []
    base_port = 5300

    for i, proj_dir in enumerate(project_dirs):
        port_orig = base_port + i * 2
        port_gen = base_port + i * 2 + 1

        vfs_per_app, details = eval_project_cli(proj_dir, port_orig, port_gen)
        if vfs_per_app is not None:
            all_results.append({
                "project": proj_dir.name,
                "vfs_per_app": vfs_per_app,
                "page_details": details,
            })

    # Summary
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)

    for r in all_results:
        print(f"{r['project']:40s} VFS={r['vfs_per_app']}")

    if all_results:
        avg = sum(r["vfs_per_app"] for r in all_results) / len(all_results)
        scores = [r["vfs_per_app"] for r in all_results]
        print(f"\nAverage: {avg:.1f}  Range: [{min(scores):.1f}, {max(scores):.1f}]")

    out_path = Path(__file__).resolve().parent / f"vfs_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    out_path.write_text(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "n_projects": len(all_results),
        "results": all_results,
    }, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()


# ─── OCR fallback for input PNGs ─────────────────────────────────

def _ocr_input_blocks(png_path, viewport_w=1440, viewport_h=900):
    """Fallback: OCR input PNG into pseudo DOM blocks. Used when reference
    dev server can't render the route (auth-locked, missing deps, etc).
    Returns blocks in the same shape as `_EXTRACT_BLOCKS_JS` produces, so
    `compute_vfs_page` can consume them directly.

    Each OCR'd line becomes one block with text + normalised bbox. Colour and
    tag are placeholders (CIEDE2000 colour comparison is meaningless without
    real foreground colour info, but Text/Position/Size still work).
    """
    try:
        from PIL import Image
        import pytesseract
    except ImportError:
        return []
    try:
        img = Image.open(png_path)
        w, h = img.size
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    except Exception as e:
        print(f"    [OCR] failed for {Path(png_path).name}: {str(e)[:60]}")
        return []

    # Group words into lines using (block_num, par_num, line_num)
    lines = {}
    n = len(data.get("text", []))
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        try:
            conf = float(data.get("conf", [0]*n)[i])
        except Exception:
            conf = 0
        if conf < 30:  # skip low-confidence words
            continue
        key = (data.get("block_num",[0]*n)[i],
               data.get("par_num",[0]*n)[i],
               data.get("line_num",[0]*n)[i])
        l = data["left"][i]; t = data["top"][i]
        r = l + data["width"][i]; b = t + data["height"][i]
        cur = lines.get(key)
        if cur is None:
            lines[key] = {"words":[txt], "left":l, "top":t, "right":r, "bottom":b}
        else:
            cur["words"].append(txt)
            cur["left"] = min(cur["left"], l)
            cur["top"] = min(cur["top"], t)
            cur["right"] = max(cur["right"], r)
            cur["bottom"] = max(cur["bottom"], b)

    blocks = []
    for info in lines.values():
        text = " ".join(info["words"]).lower()[:200]
        if len(text) < 2:
            continue
        blocks.append({
            "text": text,
            "bbox": {"x": info["left"]/w, "y": info["top"]/h,
                     "w": (info["right"]-info["left"])/w,
                     "h": (info["bottom"]-info["top"])/h},
            "fg_color": [0, 0, 0],  # placeholder — Color metric degraded for OCR-derived blocks
            "tag": "p",
        })
    return blocks
