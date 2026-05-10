"""
UI2App route-matching cascade.

Stage 4 Coverage orchestrator (compute_coverage) plus cascade fallback tiers T1-T5b
(paper Appendix VFS Cascade):
  T1 exact route match
  T2 dynamic-route regex
  T3 path-suffix
  T4 synonym structural
  T5a phash visual-first Hungarian
  T5b legacy SSIM+hist

DOM-alignment default path: see _dom_align_match.
"""

import json, os, re, subprocess, sys, hashlib, time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from support.server import start_server, stop_server
from support.blocks import extract_blocks, compute_vfs_page, _ocr_input_blocks
from support.auth import (detect_gates, _try_dismiss_gate, detect_localstorage_seeds,
                       _apply_localstorage_seeds, detect_inmemory_auth_pattern,
                       _attempt_form_login, _spa_goto)
from stages.generate import extract_test_routes
from stages.visual import _is_rendered, _gated_extract_blocks


# ─── Visual-first matching (phash) ─────────────────────────────────

# phash floor for accepting a visual-first match. phash hamming distance on a
# 16×16 hash gives 256 bits, so similarity = 1 - hamming/256. Pairs that share
# layout AND colour palette typically score 0.55+; pairs sharing only one of
# them score 0.40-0.55; truly different pages score below 0.40.
PHASH_FLOOR = 0.45

def _phash_similarity(path_a, path_b, hash_size=16):
    """phash-based similarity in [0, 1]. Robust to small layout shifts and
    stylistic differences (font, padding) that wreck SSIM on dark/icon-dense
    web pages. Falls back to 0.0 on any error so a missing file or unreadable
    image silently degrades to "no match" rather than crashing."""
    try:
        import imagehash
        from PIL import Image
        h_a = imagehash.phash(Image.open(path_a), hash_size=hash_size)
        h_b = imagehash.phash(Image.open(path_b), hash_size=hash_size)
        max_bits = hash_size * hash_size
        return max(0.0, 1.0 - (h_a - h_b) / max_bits)
    except Exception as e:
        return 0.0


def _visual_first_match(inputs_with_dirs, rendered, max_per_route, abs_floor=PHASH_FLOOR):
    """Hungarian-based visual-first matcher.

    Builds an M×N phash similarity matrix between inputs and rendered gen
    routes, then solves the assignment globally with `linear_sum_assignment`.
    Greedy is unsuitable here: when phash similarities cluster in 0.5-0.65
    across diverse pages (typical for dark-theme/icon-heavy apps), greedy
    locks high-sim pairs early and forces later inputs into worse slots —
    Hungarian considers the full cost matrix and finds the globally lowest
    total cost assignment.

    Route reuse is handled by column duplication: each concrete route gets a
    single column, each dynamic route (`:id`) gets `max_per_route` copies so
    the same route can absorb multiple inputs. After assignment, pairs with
    similarity below `abs_floor` are dropped (the input becomes MISS).

    Returns list of (ss_dict, route_def, gen_path, sim) tuples.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    gen_available = [(rd, gp) for rd, gp in rendered.items()
                     if isinstance(gp, str) and os.path.isfile(gp)]
    if not gen_available or not inputs_with_dirs:
        return []

    # All routes get exactly one slot in visual-first matching: phash similarity
    # alone is too noisy to justify the *same* gen page being assigned as the
    # visual best to several different-looking inputs. Legitimate dynamic-route
    # multi-matches (e.g., product-1 and product-5 both → /product/:id) are
    # handled by cascade T2 in Phase 1 / Phase 3 where the input route's
    # path-name evidence is what justifies the reuse.
    columns = [(rd, gp) for rd, gp in gen_available]

    n_in = len(inputs_with_dirs)
    n_col = len(columns)
    sim_matrix = np.zeros((n_in, n_col))
    for i, (ss, ss_dir) in enumerate(inputs_with_dirs):
        input_path = os.path.join(str(ss_dir), ss["filename"])
        if not os.path.isfile(input_path):
            continue
        for j, (rd, gp) in enumerate(columns):
            sim_matrix[i][j] = _phash_similarity(input_path, gp)

    # Pad to square with cost 1 (= sim 0) so unfilled assignments are no-cost
    # to drop after the absolute floor.
    cost = 1.0 - sim_matrix
    size = max(n_in, n_col)
    padded = np.ones((size, size))
    padded[:n_in, :n_col] = cost
    row_ind, col_ind = linear_sum_assignment(padded)

    matched = []
    for i, j in zip(row_ind, col_ind):
        if i >= n_in or j >= n_col:
            continue
        sim = float(sim_matrix[i][j])
        if sim < abs_floor:
            continue
        ss, _ = inputs_with_dirs[i]
        rd, gp = columns[j]
        matched.append((ss, rd, gp, round(sim, 3)))
    return matched



# ─── dom_align DOM-block alignment matching ─────────────────────────────

# Absolute floor on DOM-block alignment score (page VFS / 100). 0.20 means a
# pair is accepted if the Hungarian text-block alignment between reference and
# gen DOM yields a page VFS of 20+. Below this, the input is treated as MISS —
# the model didn't produce content that could plausibly be the same page.
DOM_FLOOR = 0.20

import filelock as _filelock_mod

def _dom_align_match(ss_dir, manifest, source_dir, source_port,
                  rendered, gen_blocks_cache, gate_info, gen_routes,
                  ref_blocks_cache, record_match_fn):
    """dom_align matching: build M×N reference-vs-gen DOM block alignment matrix and
    Hungarian-assign. Mutates `ref_blocks_cache` in place with extracted blocks.

    Returns True if matching ran successfully (matched_pairs/coverage_details
    populated by record_match_fn callbacks); False if source server failed
    and caller should fall back to cascade.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from support.blocks import compute_vfs_page

    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        print(f"  [Cov-domalign] source_dir {source_dir} missing — falling back")
        return False

    # Rendered routes only (gen blocks must exist).
    available_routes = [(rd, gen_blocks_cache.get(rd, [])) for rd, gp in rendered.items()
                        if rd not in ("__gate_info__", "__blocks_cache__")
                        and isinstance(gp, str) and os.path.isfile(gp)]
    if not available_routes:
        print(f"  [Cov-domalign] no rendered gen routes — falling back")
        return False

    if (source_dir / "package.json").exists() and not (source_dir / "node_modules").exists():
        print(f"  [Cov-domalign] installing source deps...")
        subprocess.run(["pnpm", "install"], cwd=str(source_dir), capture_output=True, timeout=600)

    # Lock source dir against concurrent dev-server starts (multiple models eval
    # the same source project — same lock that compute_vfs_dom uses).
    lock = _filelock_mod.FileLock(source_dir / ".ui2app_vfs.lock", timeout=300)
    lock.acquire()
    print(f"  [Cov-domalign] acquired source lock on {source_dir.name}")

    src_srv, src_url = start_server(source_dir, source_port)
    if src_url is None:
        print(f"  [Cov-domalign] source server failed to start — falling back")
        stop_server(src_srv)
        lock.release()
        return False

    # Seed source-side auth state too. Reference apps also commonly gate
    # `/dashboard` etc behind a useAuth check that reads localStorage; without
    # seeding, every reference extraction at non-/login routes would walk the
    # Login form's DOM rather than the real route DOM.
    src_seeds = detect_localstorage_seeds(source_dir)
    if src_seeds:
        print(f"  [Cov-domalign] localStorage auth seed (ref): {[k for k,_ in src_seeds]}")

    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1440, "height": 900})
            _apply_localstorage_seeds(context, src_seeds)
            page = context.new_page()

            # Step 1: extract reference blocks per input
            print(f"  [Cov-domalign] extracting reference DOM blocks for {len(manifest['screenshots'])} inputs")
            for ss in manifest["screenshots"]:
                route = ss.get("route", "/" + ss.get("label", ""))
                ref_url = src_url + route.split("?")[0]
                blocks = _gated_extract_blocks(page, ref_url, gate_info=None,
                                               wait_ms=4000, timeout=30000,
                                               route_path=route)
                # OCR fallback when ref dev server returns nothing — typically
                # because the route is auth-locked (NextAuth cookie session,
                # server-side `if (!user) redirect("/login")`, etc.) and our
                # localStorage seeding can't bypass it. The original input PNG
                # was captured from a state where the human/session held the
                # right cookies, so OCR'ing it gives back the same content
                # that the reference WOULD render if we could reach it.
                source = "DOM"
                if not blocks:
                    input_path = os.path.join(str(ss_dir), ss["filename"])
                    ocr_blocks = _ocr_input_blocks(input_path)
                    if ocr_blocks:
                        blocks = ocr_blocks
                        source = "OCR"
                ref_blocks_cache[ss["filename"]] = blocks
                tag = "REF-OCR" if source == "OCR" else "REF    "
                print(f"    {tag} {ss['filename']:35s} {route:30s} blocks={len(blocks)} (source={source})")

            browser.close()
    finally:
        stop_server(src_srv)
        lock.release()

    # Step 2: build M×N similarity matrix using compute_vfs_page
    inputs = manifest["screenshots"]
    M = len(inputs)
    N = len(available_routes)
    sim_matrix = np.zeros((M, N))
    for i, ss in enumerate(inputs):
        ref_b = ref_blocks_cache.get(ss["filename"], [])
        for j, (rd, gen_b) in enumerate(available_routes):
            if not ref_b or not gen_b:
                continue
            page_vfs, _ = compute_vfs_page(ref_b, gen_b)
            sim_matrix[i][j] = page_vfs / 100.0

    # Token-overlap boost: pairs whose route paths share lexical tokens get
    # a small additive bump. Keeps DOM-align in charge globally but breaks
    # ties (e.g., `/leads` ↔ `/leads` should win over `/leads` ↔ `/settings/team`
    # when DOM scores are within ~0.05 of each other).
    TOKEN_BONUS = 0.08
    SPLIT_RE = re.compile(r"[-_/]+")
    def _tokens(path):
        return set(t for t in SPLIT_RE.split(path.lower().strip("/")) if t and t not in ("img",))
    boosted = sim_matrix.copy()
    for i, ss in enumerate(inputs):
        in_route = ss.get("route", "/" + ss.get("label", "")).lower().strip("/")
        in_label = ss.get("label", "").lower().strip()
        in_toks = _tokens(in_route) | _tokens(in_label)
        for j, (rd, _) in enumerate(available_routes):
            if rd == "__login__":
                gen_toks = {"login", "signin", "auth"}
            else:
                gen_toks = _tokens(rd)
            if in_toks and gen_toks:
                jacc = len(in_toks & gen_toks) / len(in_toks | gen_toks)
                boosted[i][j] = min(1.0, sim_matrix[i][j] + TOKEN_BONUS * jacc)

    cost = 1.0 - boosted
    size = max(M, N)
    padded = np.ones((size, size))
    padded[:M, :N] = cost
    row_ind, col_ind = linear_sum_assignment(padded)

    print(f"  [Cov-domalign] DOM-alignment Hungarian (floor={DOM_FLOOR}, +token_boost={TOKEN_BONUS})")
    for i, j in zip(row_ind, col_ind):
        if i >= M or j >= N:
            continue
        sim = float(sim_matrix[i][j])
        if sim < DOM_FLOOR:
            continue
        ss = inputs[i]
        rd, _ = available_routes[j]
        gp = rendered[rd]
        input_path = os.path.join(str(ss_dir), ss["filename"])
        phash_side = _phash_similarity(input_path, gp)
        boost = float(boosted[i][j] - sim_matrix[i][j])
        record_match_fn(ss, rd, gp, "dom_align", phash_side)
        print(f"    DMATCH  {ss['filename']:35s} → {rd}  (dom_align={sim:.3f}{f', +tok={boost:.3f}' if boost > 0 else ''}, phash={phash_side:.3f})")

    return True



# ─── Stage 4: Coverage ─────────────────────────────────────────

SYNONYMS = [
    {"product", "products", "item", "detail", "single"},
    {"category", "categories", "cat"},
    # NOTE: "dashboard" removed — for admin apps `/` is often landing while
    # the real dashboard lives at `/app/dashboard` or `/dashboard`.
    # Bundling them causes /dashboard inputs to wrongly match `/`.
    {"home", "landing", "index", "overview"},
    {"blog", "blogs", "post", "posts", "article", "articles", "writing"},
    {"about", "info"},
    {"search", "find", "result", "results"},
    {"signin", "sign-in", "login", "auth", "authentication",
     "signup", "sign-up", "register", "create",
     "forgot", "forgot-password", "reset", "reset-password",
     "otp", "2fa", "verify", "verification"},
    {"project", "projects", "archive", "portfolio", "work"},
    {"profile", "account", "member", "members", "user", "users"},
    {"setting", "settings", "preference", "preferences", "config"},
    {"chat", "chats", "message", "messages", "messaging"},
    # "etc" / "misc" / "links" / "elsewhere" are common author idioms for a
    # miscellaneous links / contact page (e.g., Phineas portfolio's `/etc` is a
    # social-links page that maps to a model's `/more-contact` form).
    {"contact", "support", "etc", "misc", "links", "elsewhere"},
    {"responsive", "mobile"},
    {"table", "tables"},
    {"form", "forms"},
    {"ui", "element", "elements", "component", "components"},
    {"video", "videos", "media"},
    {"tag", "tags", "label", "labels"},
    {"guide", "guides", "doc", "docs", "document", "documents", "documentation"},
    {"chart", "charts", "graph", "graphs", "analytics"},
    {"order", "orders"},
    {"alert", "alerts", "notification", "notifications"},
    {"button", "buttons"},
    {"image", "images", "photo", "photos", "gallery"},
    {"modal", "modals", "dialog", "dialogs"},
    {"badge", "badges"},
    {"avatar", "avatars"},
    {"icon", "icons"},
    {"calendar", "schedule"},
    {"pricing", "price", "prices", "plan", "plans"},
    {"faq", "faqs", "help"},
]


def _expand_keywords(kws):
    expanded = set(kws)
    for kw in kws:
        for group in SYNONYMS:
            if kw in group:
                expanded |= group
    return expanded


def match_input_to_route(input_label, input_route, gen_routes_list, route_usage=None, max_per_route=4):
    if route_usage is None:
        route_usage = {}
    input_route_clean = input_route.split("?")[0]
    input_route_norm = input_route_clean.lower().strip("/")
    input_label_norm = input_label.lower().strip()

    # Strict 1-to-1 across BOTH concrete and dynamic routes (cascade). Each rendered
    # gen page is paired with at most one input. The previous policy of letting
    # dynamic routes (`/post/:id`, `/product/:id`) absorb up to `max_per_route`
    # inputs sounded right for the shopco product-1/product-5 case but in
    # practice (Qwen3.5 genshin) `/post/:id` ended up consuming 3-5 unrelated
    # inputs (zen-keyboard, blog, blog-posts, keybinds, theme) inflating Coverage to
    # 9/9 against only 5 unique gen renderings. Strict cap=1 gives a single
    # well-defined "this gen page best matches this one input" pairing; the
    # legitimate two-product case correspondingly drops one input to MISS.
    def _can_use(rd):
        return route_usage.get(rd, 0) < 1
    def _consume(rd):
        route_usage[rd] = route_usage.get(rd, 0) + 1

    # Exact match
    for gr in gen_routes_list:
        gr_url = gr["url"].lower().strip("/")
        gr_def = gr["def"].lower().strip("/")
        if input_route_norm == gr_url or input_route_norm == gr_def:
            if _can_use(gr["def"]):
                _consume(gr["def"])
                return gr

    # Dynamic route match
    for gr in gen_routes_list:
        gr_def = gr["def"]
        if ":" not in gr_def:
            continue
        if not _can_use(gr_def):
            continue
        pattern = re.sub(r':(\w+)', r'[^/]+', gr_def.lower().strip("/"))
        if re.fullmatch(pattern, input_route_norm):
            _consume(gr_def)
            return gr

    # Path-suffix match: input "/dashboard" ↔ gen "/app/dashboard".
    # Also handles prefix-mismatch cases:
    #   input "/dashboard/customers/import-export" ↔ gen "/app/customers/import-export"
    # by finding the longest trailing-segment overlap. Prefer most-specific match.
    if input_route_norm:
        input_segs = input_route_norm.split("/")
        best, best_score = None, 0
        for gr in gen_routes_list:
            if not _can_use(gr["def"]): continue
            gr_path = gr["def"].lower().strip("/")
            if not gr_path: continue
            gr_segs = gr_path.split("/")
            # Find longest common suffix between input_segs and gr_segs
            k = 0
            while k < min(len(input_segs), len(gr_segs)) and \
                  input_segs[-(k+1)] == gr_segs[-(k+1)]:
                k += 1
            # Score: longer overlap wins; ties broken by gen route specificity
            if k >= 1 and k > best_score:
                best, best_score = gr, k
        if best is not None:
            _consume(best["def"])
            return best

    # Synonym-based structural match
    input_kw = set(re.split(r'[-_/]', input_label_norm)) - {"", "img"}
    input_kw |= set(re.split(r'[-_/]', input_route_norm)) - {""}
    input_kw |= set(re.split(r'[_/]', input_label_norm)) - {"", "img"}
    input_kw |= set(re.split(r'[_/]', input_route_norm)) - {""}
    input_expanded = _expand_keywords(input_kw)

    for gr in gen_routes_list:
        if not _can_use(gr["def"]):
            continue
        gr_base = re.sub(r'/:[^/]+', '', gr["def"]).strip("/")
        gr_kw = set(re.split(r'[-_/]', gr_base.lower())) - {""}
        if not gr_kw and gr["def"].strip("/") == "":
            gr_kw = {"home"}
        gr_expanded = _expand_keywords(gr_kw)
        overlap = input_expanded & gr_expanded
        if len(overlap) >= 1 and (gr_expanded & input_kw or gr_kw & input_expanded):
            _consume(gr["def"])
            return gr

    return None



# ─── Visual Fallback Matching ────────────────────────────────────

def _visual_similarity(img_path_a, img_path_b):
    """Compute structural similarity between two screenshots.

    Uses a combination of:
      1. SSIM on resized grayscale (structural layout)
      2. Histogram correlation (color/tone distribution)
    Returns score in [0, 1] where 1 = identical.
    """
    try:
        from PIL import Image
        import numpy as np

        size = (256, 160)  # landscape ratio, enough detail for layout
        img_a = np.array(Image.open(img_path_a).convert("RGB").resize(size, Image.LANCZOS), dtype=np.float32)
        img_b = np.array(Image.open(img_path_b).convert("RGB").resize(size, Image.LANCZOS), dtype=np.float32)

        # --- SSIM (simplified, per-channel mean) ---
        def _ssim_channel(a, b):
            C1, C2 = 6.5025, 58.5225  # (0.01*255)^2, (0.03*255)^2
            mu_a, mu_b = a.mean(), b.mean()
            var_a, var_b = a.var(), b.var()
            cov = ((a - mu_a) * (b - mu_b)).mean()
            num = (2 * mu_a * mu_b + C1) * (2 * cov + C2)
            den = (mu_a**2 + mu_b**2 + C1) * (var_a + var_b + C2)
            return float(num / den)

        ssim = np.mean([_ssim_channel(img_a[:,:,c], img_b[:,:,c]) for c in range(3)])

        # --- Histogram correlation ---
        def _hist_corr(a, b):
            ha = np.histogram(a, bins=64, range=(0, 256))[0].astype(np.float32)
            hb = np.histogram(b, bins=64, range=(0, 256))[0].astype(np.float32)
            ha = ha / (ha.sum() + 1e-8)
            hb = hb / (hb.sum() + 1e-8)
            return float(np.dot(ha, hb) / (np.linalg.norm(ha) * np.linalg.norm(hb) + 1e-8))

        hist = np.mean([_hist_corr(img_a[:,:,c], img_b[:,:,c]) for c in range(3)])

        # Weighted combination: SSIM dominates (layout), histogram is secondary (tone)
        return 0.7 * max(ssim, 0) + 0.3 * max(hist, 0)
    except Exception as e:
        print(f"    [VisualSim] ERROR: {e}")
        return 0.0


MIN_VISUAL_SIM = 0.55  # absolute floor for T5 fallback acceptance

def _visual_fallback_match(unmatched_inputs, rendered, route_usage, max_per_route,
                           margin_ratio=1.2, min_sim=MIN_VISUAL_SIM):
    """Try to match unmatched input screenshots to rendered gen screenshots via visual similarity.

    Two thresholds must hold simultaneously:
      (a) relative: `best >= mean * margin_ratio` (avoids ambiguous picks when the
          candidate set has uniformly low similarity);
      (b) absolute: `best >= min_sim` (rejects matches that are merely the least bad
          among a uniformly poor candidate set — e.g. a `zen-keyboard` screenshot
          paired to `/player` at SSIM+hist=0.45).

    Args:
        unmatched_inputs: list of (ss_dict, ss_dir) for unmatched input screenshots
        rendered: dict {route_def: screenshot_path} from Coverage rendering
        route_usage: current route usage counts (mutated in place)
        max_per_route: max times a gen route can be reused (concrete routes
            are additionally hard-capped at 1 per `match_input_to_route`)
        margin_ratio: best must be >= mean * margin_ratio to accept
        min_sim: absolute floor; candidates below this are rejected outright

    Returns:
        list of (ss_dict, matched_route_def, matched_gen_screenshot, similarity_score)
    """
    if not unmatched_inputs or not rendered:
        return []

    # Collect available gen screenshots
    gen_available = {}
    for route_def, gen_path in rendered.items():
        if gen_path and os.path.isfile(gen_path):
            gen_available[route_def] = gen_path

    if not gen_available or len(gen_available) < 2:
        return []

    # For each unmatched input, compute similarity to all gen screenshots
    per_input = {}  # input_filename -> [(sim, route_def, gen_path), ...]
    for ss, ss_dir in unmatched_inputs:
        input_path = os.path.join(str(ss_dir), ss["filename"])
        if not os.path.isfile(input_path):
            continue
        scores = []
        for route_def, gen_path in gen_available.items():
            sim = _visual_similarity(input_path, gen_path)
            scores.append((sim, route_def, gen_path))
        if scores:
            per_input[ss["filename"]] = (ss, scores)

    # Build candidate list: must satisfy both the relative margin and the absolute floor.
    candidates = []
    for fname, (ss, scores) in per_input.items():
        scores.sort(key=lambda x: -x[0])
        best_sim, best_route, best_path = scores[0]
        mean_sim = sum(s[0] for s in scores) / len(scores)
        rel_ok = mean_sim > 0 and best_sim >= mean_sim * margin_ratio
        abs_ok = best_sim >= min_sim
        if rel_ok and abs_ok:
            candidates.append((best_sim, ss, best_route, best_path))

    # Greedy matching: sort by similarity descending, assign best non-conflicting pairs.
    # Concrete-route reuse cap is enforced by re-checking the dynamic-vs-concrete
    # rule already encoded in `match_input_to_route._can_use`.
    candidates.sort(key=lambda x: -x[0])
    matched = []
    used_inputs = set()
    for sim, ss, route_def, gen_path in candidates:
        input_key = ss["filename"]
        if input_key in used_inputs:
            continue
        if route_usage.get(route_def, 0) >= 1:  # cascade strict 1-to-1
            continue
        matched.append((ss, route_def, gen_path, round(sim, 3)))
        used_inputs.add(input_key)
        route_usage[route_def] = route_usage.get(route_def, 0) + 1

    return matched


_RENDER_PROBE_JS = """
(root) => {
    const all = root.querySelectorAll('*').length;
    const text = (root.innerText || '').trim().length;
    const visuals = root.querySelectorAll('svg, img, canvas, video').length;
    const interactive = root.querySelectorAll('button, input, textarea, select, a[href], [role="button"]').length;
    return { all, text, visuals, interactive };
}
"""


def compute_coverage(ss_dir, gen_dir, port=5280, source_dir=None, source_port=5295):
    """Compute Stage 4 Coverage (route matching). Returns (coverage_score, matched_pairs, coverage_details, rendered, unmatched_gen_routes, empty_gen_routes).

    When `source_dir` is provided, dom_align DOM-block alignment matching is used
    instead of cascade + visual-first phash. The pipeline:
      1. Render each gen route, extract its DOM text blocks (cached for VFS reuse).
      2. Render each input route on the source dev server, extract its DOM blocks.
      3. Build M×N similarity matrix where sim[i][j] = compute_vfs_page(ref_i, gen_j) / 100.
      4. Hungarian assignment (cap=1 for all routes) with absolute floor `DOM_FLOOR`.
    DOM-text matching identifies cases where the model implemented the right
    *content* under a different *route name* (e.g., gen `/` actually renders
    Blog content that should pair with the reference's `/blog` input), and
    rejects path-name matches whose content has diverged (gen `/` is Blog but
    reference `/` is a music landing).

    When `source_dir` is None, falls back to cascade cascade-first + visual-first.
    """
    manifest = json.loads((ss_dir / "manifest.json").read_text())
    M = len(manifest["screenshots"])
    gen_routes = extract_test_routes(gen_dir)
    eval_dir = gen_dir.parent / "eval_screenshots"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Detect gate components (useState early-return, redirect, etc.) so we can
    # dismiss them before each route screenshot — otherwise every gated route
    # captures the same intro/login/splash screen instead of its real component.
    gate_info = detect_gates(gen_dir)
    if gate_info["has_gate"]:
        print(f"  [Cov] gate detected: {gate_info['pattern']} -> {gate_info.get('gate_component','?')}; "
              f"dismiss selectors: {gate_info['dismiss_selectors']}")

    srv, url = start_server(gen_dir, port)
    rendered = {}
    if url is None:
        print(f"  [Cov] Server failed to start, skipping route rendering")
        stop_server(srv)
        return 0.0, [], {}, {}, [], []
    # Order routes so `/` renders before its potential redirect target — this
    # makes `gen_home.png` the canonical file when `<Route path="/" element={<Navigate to="/foo">}>`
    # collapses `/` and `/foo` to the same DOM.
    gen_routes_sorted = sorted(gen_routes, key=lambda r: (r["url"] != "/", len(r["url"]), r["url"]))
    seen_dom_hashes = {}   # dom_hash -> (route_def, screenshot_path)
    aliases = {}           # alias_route_def -> canonical_route_def
    gen_blocks_cache = {}  # route_def -> [block dicts] — fed into dom_align cost matrix and VFS reuse

    # Seed gen-side auth state before app boots — keeps useAuth-style gates
    # from intercepting every route with the Login form (kimi-k2.5 daisyUI
    # case: 9 routes all rendering the same Login form pre-fix).
    gen_seeds = detect_localstorage_seeds(gen_dir)
    if gen_seeds:
        print(f"  [Cov] localStorage auth seed (gen): {[k for k,_ in gen_seeds]}")
    # Detect in-memory React Context auth — localStorage seeding can't bypass
    # this; we'll instead form-fill /login on a single page and reuse it for
    # all subsequent routes via SPA navigation (history.pushState).
    inmemory_auth = (not gen_seeds) and detect_inmemory_auth_pattern(gen_dir)
    if inmemory_auth:
        print(f"  [Cov] in-memory React auth detected (no localStorage); will attempt form-fill recovery")

    try:
        from playwright.sync_api import sync_playwright
        from support.blocks import _EXTRACT_BLOCKS_JS
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1440, "height": 900})
            _apply_localstorage_seeds(context, gen_seeds)
            page = context.new_page()

            # In-memory auth recovery: form-fill /login once, then use SPA
            # navigation for the rest of the loop so React state survives.
            spa_mode = False
            if inmemory_auth:
                # Capture pre-auth Login form FIRST so login-labeled inputs
                # have a legitimate match candidate (`__login__` synthetic
                # route). Without this, all routes post form-fill render
                # post-auth content and login input gets mis-assigned to a
                # random dashboard page.
                login_route = next((tr for tr in gen_routes_sorted
                                    if any(tok in tr["def"].lower()
                                           for tok in ("/login", "/signin", "/sign-in", "/auth/login"))),
                                   None)
                if login_route is not None:
                    try:
                        page.goto(f"{url}{login_route['url']}", wait_until="domcontentloaded", timeout=10000)
                        page.wait_for_timeout(2500)
                        root = page.query_selector("#root")
                        probe = root.evaluate(_RENDER_PROBE_JS) if root else {"all":0,"text":0,"visuals":0,"interactive":0}
                        if _is_rendered(probe):
                            sp_login = str(eval_dir / "gen_login_pre_auth.png")
                            page.screenshot(path=sp_login)
                            rendered["__login__"] = sp_login
                            try:
                                gen_blocks_cache["__login__"] = page.evaluate(_EXTRACT_BLOCKS_JS) or []
                            except Exception:
                                gen_blocks_cache["__login__"] = []
                            print(f"    RENDER  __login__ (pre-auth, in-memory)    → gen_login_pre_auth.png  (text={probe['text']} blocks={len(gen_blocks_cache['__login__'])})")
                    except Exception as e:
                        print(f"    [Cov] pre-auth Login capture failed: {str(e)[:60]}")
                ok = _attempt_form_login(page, url, gen_routes_sorted, rendered)
                if ok:
                    spa_mode = True
                    print(f"  [Cov] form-fill login succeeded; switching to SPA navigation")
                else:
                    print(f"  [Cov] form-fill login failed; falling back to per-route goto")

            for tr in gen_routes_sorted:
                try:
                    if spa_mode:
                        # SPA pushState — keeps React in-memory auth flag alive
                        if not _spa_goto(page, tr["url"]):
                            page.goto(f"{url}{tr['url']}", wait_until="domcontentloaded", timeout=10000)
                    else:
                        page.goto(f"{url}{tr['url']}", wait_until="domcontentloaded", timeout=10000)
                    page.wait_for_timeout(2000)
                    # Asymmetric dismiss policy: skip dismissal on `/` so that the
                    # gate (intro/splash) is captured for direct comparison with
                    # the reference's home screenshot, which is itself captured at
                    # `/` and therefore typically also shows the gate state. For
                    # any non-`/` route we dismiss because the reference is
                    # captured at that route directly without going through `/`.
                    is_home = tr["url"].rstrip("/") in ("", "/")
                    if gate_info["has_gate"] and not is_home:
                        _try_dismiss_gate(page, gate_info["dismiss_selectors"])
                    page.wait_for_timeout(1500)
                    root = page.query_selector("#root")
                    if root:
                        probe = root.evaluate(_RENDER_PROBE_JS)
                    else:
                        probe = {"all": 0, "text": 0, "visuals": 0, "interactive": 0}
                    if not _is_rendered(probe):
                        rendered[tr["def"]] = None
                        print(f"    EMPTY   {tr['def']:30s}  (text={probe['text']} svg/img={probe['visuals']} ui={probe['interactive']} all={probe['all']})")
                        continue

                    # DOM fingerprint: hash the route's text content + tag count
                    # signature. If a previous route already produced the same
                    # fingerprint, the current route is a redirect/alias and we
                    # reuse the prior screenshot rather than writing a duplicate.
                    fingerprint = root.evaluate(
                        "el => (el.innerText || '').trim() + '|' + el.querySelectorAll('*').length"
                    )
                    dom_hash = hashlib.md5(fingerprint.encode("utf-8")).hexdigest()
                    if dom_hash in seen_dom_hashes:
                        canonical_def, canonical_sp = seen_dom_hashes[dom_hash]
                        rendered[tr["def"]] = canonical_sp
                        aliases[tr["def"]] = canonical_def
                        print(f"    ALIAS   {tr['def']:30s} → {canonical_def}  (same DOM as already-rendered route)")
                        continue

                    fname = f"gen_{tr['url'].strip('/').replace('/', '-') or 'home'}.png"
                    sp = str(eval_dir / fname)
                    page.screenshot(path=sp)
                    rendered[tr["def"]] = sp
                    seen_dom_hashes[dom_hash] = (tr["def"], sp)
                    # Cache DOM blocks for dom_align matching and VFS reuse — same
                    # extraction the VFS stage does, hoisted here so we only pay
                    # once per gen route across the whole run.
                    try:
                        gen_blocks_cache[tr["def"]] = page.evaluate(_EXTRACT_BLOCKS_JS) or []
                    except Exception:
                        gen_blocks_cache[tr["def"]] = []
                    print(f"    RENDER  {tr['def']:30s} → {fname}  (text={probe['text']} svg/img={probe['visuals']} ui={probe['interactive']} blocks={len(gen_blocks_cache[tr['def']])})")
                except Exception as e:
                    rendered[tr["def"]] = None
                    print(f"    ERROR   {tr['def']:30s}  ({str(e)[:60]})")

            # ── Bonus pre-auth render ──
            # When we seeded auth state, all rendered pages are post-auth views.
            # Inputs labeled login/signin/etc need a Login-form candidate to
            # match against. Open a fresh non-seeded context, visit `/`, and
            # capture the gate's natural pre-auth render as a synthetic route
            # under key `__login__`.
            if gen_seeds:
                try:
                    pre_ctx = browser.new_context(viewport={"width": 1440, "height": 900})
                    pre_page = pre_ctx.new_page()
                    pre_page.goto(f"{url}/", wait_until="domcontentloaded", timeout=10000)
                    pre_page.wait_for_timeout(2500)
                    root = pre_page.query_selector("#root")
                    probe = root.evaluate(_RENDER_PROBE_JS) if root else {"all": 0, "text": 0, "visuals": 0, "interactive": 0}
                    if _is_rendered(probe):
                        sp = str(eval_dir / "gen_login_pre_auth.png")
                        pre_page.screenshot(path=sp)
                        rendered["__login__"] = sp
                        try:
                            gen_blocks_cache["__login__"] = pre_page.evaluate(_EXTRACT_BLOCKS_JS) or []
                        except Exception:
                            gen_blocks_cache["__login__"] = []
                        print(f"    RENDER  __login__ (pre-auth)             → gen_login_pre_auth.png  (text={probe['text']} blocks={len(gen_blocks_cache['__login__'])})")
                    pre_ctx.close()
                except Exception as e:
                    print(f"    [Cov] pre-auth render failed: {str(e)[:60]}")

            browser.close()
    finally:
        stop_server(srv)

    route_usage = {}
    matched_pairs = []
    coverage_details = {}
    used_filenames = set()
    ref_blocks_cache = {}  # input_filename -> ref DOM blocks (dom_align only; for VFS reuse)

    def _record_match(ss, route_def, gen_path, match_type, phash_sim, gen_url=None, route=None):
        gen_route_obj = next((r for r in gen_routes if r["def"] == route_def), None)
        if gen_url is None:
            gen_url = gen_route_obj["url"] if gen_route_obj else route_def
        if route is None:
            route = ss.get("route", "/" + ss.get("label", ""))
        matched_pairs.append({
            "input": os.path.join(str(ss_dir), ss["filename"]),
            "input_name": ss["filename"],
            "input_route": route,
            "gen_route": route_def,
            "gen_url": gen_url,
            "generated": gen_path,
        })
        coverage_details[ss["filename"]] = {
            "matched": True, "gen_route": route_def,
            "match_type": match_type, "phash_sim": round(phash_sim, 3),
        }
        used_filenames.add(ss["filename"])
        route_usage[route_def] = route_usage.get(route_def, 0) + 1

    # ── DOM-block alignment matching (only when source_dir is provided) ──
    dom_align_succeeded = False
    if source_dir is not None:
        dom_align_succeeded = _dom_align_match(
            ss_dir, manifest, source_dir, source_port,
            rendered, gen_blocks_cache, gate_info, gen_routes,
            ref_blocks_cache, _record_match,
        )
    if dom_align_succeeded:
        # dom_align owns matched_pairs / coverage_details; skip cascade cascade phases.
        unmatched = []
        for ss in manifest["screenshots"]:
            if ss["filename"] in used_filenames:
                continue
            coverage_details[ss["filename"]] = {"matched": False, "reason": "below_dom_floor"}
            unmatched.append(ss)
            print(f"    MISS    {ss['filename']:35s}  (below_dom_floor)")
    else:
        # ── Fallback: cascade-first + visual-first ──
        if source_dir is not None:
            print(f"  [Cov] DOM-align matching unavailable (source server failed) — falling back to cascade")
        # Cascade-first preserves SYNONYM matches like `etc → /more-contact` and
        # `how → /how-i-do-it` that visual phash would otherwise mis-assign on
        # the ambient 0.5-0.6 noise band. Visual-first then catches refactored
        # routes (model renamed `/products` → `/catalog`) that share content
        # but no token.
        print(f"  [Cov] Phase 1: full cascade T1-T4 (with strict cap=1 incl. dynamic)")
        for ss in manifest["screenshots"]:
            label = ss.get("label", ss["filename"].replace(".png", ""))
            route = ss.get("route", "/" + label)
            match = match_input_to_route(label, route, gen_routes, route_usage, max_per_route=1)
            if match and rendered.get(match["def"]):
                input_path = os.path.join(str(ss_dir), ss["filename"])
                gp = rendered[match["def"]]
                sim = _phash_similarity(input_path, gp)
                tier = "exact" if route.lower().strip("/") == match["url"].lower().strip("/") else "structural"
                _record_match(ss, match["def"], gp, f"cascade_{tier}", sim, match["url"], route)
                tag = "EMATCH" if tier == "exact" else "CMATCH"
                print(f"    {tag}  {ss['filename']:35s} → {match['def']}  ({tier}, phash={sim:.3f})")

        # Phase 2: visual-first Hungarian on residual inputs over remaining gen routes
        residual = [(ss, ss_dir) for ss in manifest["screenshots"] if ss["filename"] not in used_filenames]
        rendered_residual = {rd: gp for rd, gp in rendered.items()
                             if isinstance(gp, str) and os.path.isfile(gp)
                             and route_usage.get(rd, 0) < 1}
        print(f"  [Cov] Phase 2: visual-first phash Hungarian on {len(residual)} residual inputs over {len(rendered_residual)} gen routes (floor={PHASH_FLOOR})")
        if residual and rendered_residual:
            visual_matches = _visual_first_match(residual, rendered_residual,
                                                 max_per_route=1, abs_floor=PHASH_FLOOR)
            for ss, route_def, gen_path, sim in visual_matches:
                _record_match(ss, route_def, gen_path, "visual_phash", sim)
                print(f"    PMATCH  {ss['filename']:35s} → {route_def}  (phash={sim:.3f})")

        # Final pass: record MISS for inputs neither cascade nor visual matched
        unmatched = []
        for ss in manifest["screenshots"]:
            if ss["filename"] in used_filenames:
                continue
            coverage_details[ss["filename"]] = {"matched": False, "reason": "no_route_or_below_floor"}
            unmatched.append(ss)
            print(f"    MISS    {ss['filename']:35s}  (no_route_or_below_floor)")

        # Visual fallback for unmatched inputs (kept for backward-compat; usually no-op
        # given strict cap=1 already left no spare gen route).
        if unmatched:
            visual_matches = _visual_fallback_match(
                [(ss, ss_dir) for ss in unmatched],
                rendered, route_usage, max_per_route=M,
            )
            for ss, route_def, gen_path, sim in visual_matches:
                route = ss.get("route", "/" + ss.get("label", ""))
                gen_route = next((r for r in gen_routes if r["def"] == route_def), None)
                gen_url = gen_route["url"] if gen_route else route_def
                matched_pairs.append({
                    "input": str(ss_dir / ss["filename"]),
                    "input_name": ss["filename"],
                    "input_route": route,
                    "gen_route": route_def,
                    "gen_url": gen_url,
                    "generated": gen_path,
                })
                coverage_details[ss["filename"]] = {
                    "matched": True, "gen_route": route_def,
                    "match_type": "visual", "visual_sim": sim,
                }
                print(f"    VMATCH  {ss['filename']:35s} → {route_def}  (sim={sim:.3f})")

    coverage = round(len(matched_pairs) / max(M, 1) * 100, 1)

    # Diagnostic: gen routes that rendered successfully but no input ever paired with them.
    # These are real generated pages the model produced that the cascade could not anchor
    # to any reference screenshot (extra/refactored routes, redirect targets, etc.).
    used_gen_routes = {info["gen_route"] for info in coverage_details.values() if info.get("matched")}
    rendered_routes = {rd for rd, p in rendered.items() if p}
    unmatched_gen_routes = sorted(rendered_routes - used_gen_routes)
    empty_gen_routes = sorted(rd for rd, p in rendered.items() if not p)
    if unmatched_gen_routes:
        print(f"  [Cov] orphan gen routes (rendered, never matched): {unmatched_gen_routes}")
    if empty_gen_routes:
        print(f"  [Cov] empty gen routes (failed render heuristic): {empty_gen_routes}")

    # Stash gate diagnostics on the rendered dict via a sentinel key so
    # run_evaluation can forward it without changing the return arity again.
    rendered.setdefault("__gate_info__", {
        "has_gate": gate_info["has_gate"],
        "pattern": gate_info["pattern"],
        "gate_component": gate_info["gate_component"],
        "redirects": gate_info["redirects"],
        "dom_aliases": aliases,  # alias_route -> canonical_route (DOM-hash dedup)
    })
    # Cache DOM blocks so VFS can skip re-extraction (~halves VFS wall time when dom-align ran).
    rendered.setdefault("__blocks_cache__", {
        "gen_blocks": gen_blocks_cache,
        "ref_blocks": ref_blocks_cache,
    })
    if aliases:
        print(f"  [Cov] DOM-hash alias map: {aliases}")

    print(f"  [Cov] {len(matched_pairs)}/{M} covered (coverage={coverage})")
    return coverage, matched_pairs, coverage_details, rendered, unmatched_gen_routes, empty_gen_routes


