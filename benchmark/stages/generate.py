#!/usr/bin/env python3
"""
UI2App — Code Generation (Stage 1-3).

Stage 1: VLM plan + code generation (multi-turn, screenshots sent once)
Stage 2: Exec check (vite build + Playwright render)
Stage 3: Self-debug loop (same model, max 3 rounds)

Usage:
  from generate import run_generation
  generated, gen_dir, exec_at1, exec_at3, debug_info = run_generation(client, model, ss_dir, run_dir)
"""

import json, os, re, subprocess, sys, base64
from pathlib import Path

from support.projects import ROOT, SCAFFOLD, ENTRY_FILES
from support.llm import get_client, chat, log_cost
from stages.prompts import (
    make_plan_prompt, make_gen_prompt,
    make_locate_prompt, make_runtime_fix_prompt, make_build_fix_prompt,
)
from support.server import start_server, stop_server

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ─── Stage 1: Code Generation ─────────────────────────────────────

def format_scaffold_turn1():
    parts = ["I've set up the project scaffold with the following files:\n"]
    for path, content in SCAFFOLD.items():
        parts.append(f"--- {path} ---")
        parts.append(content)
        parts.append("")
    parts.append("The project is ready. All dependencies are installed. You can now create source files under src/.")
    return "\n".join(parts)


def _resize_if_oversized(img_path, max_dim=7500):
    from PIL import Image
    import io
    img = Image.open(img_path)
    w, h = img.size
    if w <= max_dim and h <= max_dim:
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    scale = min(max_dim / w, max_dim / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    print(f"    [Resize] {img_path.name}: {w}x{h} -> {new_w}x{new_h}")
    return base64.b64encode(buf.getvalue()).decode()


def load_screenshots_as_content(ss_dir):
    manifest = json.loads((ss_dir / "manifest.json").read_text())
    images = []
    for ss in manifest["screenshots"]:
        b64 = _resize_if_oversized(ss_dir / ss["filename"])
        images.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    # Also include nav_hints if present
    for nh in manifest.get("nav_hints", []):
        f = ss_dir / nh["filename"]
        if f.exists():
            b64 = _resize_if_oversized(f)
            images.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    total = manifest.get("total", len(manifest["screenshots"])) + len(manifest.get("nav_hints", []))
    return images, total


def generate_code(client, model, ss_dir, run_dir=None):
    images, M = load_screenshots_as_content(ss_dir)
    scaffold_turn1 = format_scaffold_turn1()

    plan_prompt = make_plan_prompt(M)

    print(f"  [Plan] {M} screenshots, model={model}...")
    messages = [
        {"role": "assistant", "content": scaffold_turn1},
        {"role": "user", "content": [*images, {"type": "text", "text": plan_prompt}]},
    ]
    plan_text, _ = chat(client, model, messages, max_tokens=4096, stage="plan")

    # Persist raw plan text BEFORE parse, so when JSON parse fails we still have
    # the model's output for later analysis / json-repair.
    if run_dir is not None:
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "plan_raw.txt").write_text(plan_text)
        except Exception:
            pass  # never block generation on logging

    json_match = re.search(r'\{[\s\S]*\}', plan_text)
    if json_match:
        plan_text = json_match.group(0)
    try:
        plan_data = json.loads(plan_text)
    except json.JSONDecodeError:
        from json_repair import repair_json
        plan_data = json.loads(repair_json(plan_text))
        print("  [Plan] ⚠ Used json-repair fallback for malformed plan JSON")
    plan = plan_data.get("plan", [])
    extra_deps = plan_data.get("extra_dependencies", {})
    print(f"  [Plan] {len(plan)} files, extra_deps: {list(extra_deps.keys())}")

    plan_summary = "\n".join(f"  - {f.get('path','?')}: {f.get('description','')}" for f in plan if isinstance(f, dict))
    gen_prompt = make_gen_prompt(plan_summary)

    messages.append({"role": "assistant", "content": plan_text})
    messages.append({"role": "user", "content": gen_prompt})
    print(f"  [Gen] Multi-turn generation ({len(plan)} files, images NOT re-sent)...")

    full_output = ""
    max_continues = 3
    for attempt in range(1 + max_continues):
        stage = "gen" if attempt == 0 else f"gen_continue_{attempt}"
        text, finish = chat(client, model, messages, max_tokens=32768, stage=stage)
        full_output += text
        if finish == "stop":
            print(f"  [Gen] Complete (finish_reason=stop)")
            break
        if attempt < max_continues:
            print(f"  [Gen] Truncated (attempt {attempt+1}), sending Continue...")
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": "Continue writing from exactly where you left off. Do not repeat any content. Continue with the next file or finish the current file."})
        else:
            print(f"  [Gen] Truncated after {max_continues} continues, using partial output")

    generated = parse_single_pass_output(full_output)
    print(f"  [Gen] Parsed {len(generated)} files from output")
    return generated, extra_deps, plan_data


def parse_single_pass_output(output):
    generated = {}
    parts = re.split(r'^---\s+(src/[^\s]+)\s+---\s*$', output, flags=re.MULTILINE)
    for i in range(1, len(parts) - 1, 2):
        fpath = parts[i].strip()
        code = parts[i + 1].strip()
        if fpath and code:
            code = re.sub(r'^```\w*\n?', '', code)
            code = re.sub(r'\n```\s*$', '', code)
            generated[fpath] = code
    if not generated:
        blocks = re.findall(r'```(?:tsx?|jsx?|typescript|javascript)?\s*\n// (src/[^\n]+)\n([\s\S]*?)```', output)
        for fpath, code in blocks:
            generated[fpath.strip()] = code.strip()
    if not generated:
        print("  [WARN] Could not parse file delimiters, attempting loose parsing...")
        blocks = re.findall(r'(src/\S+\.tsx?)\s*\n([\s\S]*?)(?=\nsrc/\S+\.tsx?|\Z)', output)
        for fpath, code in blocks:
            generated[fpath.strip()] = code.strip()
    return generated




def write_project(generated, extra_deps, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for p, c in SCAFFOLD.items():
        fp = out_dir / p
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(c)
    for p, c in generated.items():
        fp = out_dir / p
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(c)
    if extra_deps:
        pkg = json.loads((out_dir / "package.json").read_text())
        pkg["dependencies"].update(extra_deps)
        (out_dir / "package.json").write_text(json.dumps(pkg, indent=2))
    print("  [Build] pnpm install...")
    r = subprocess.run(["pnpm", "install"], cwd=str(out_dir), capture_output=True, text=True, timeout=240)
    if r.returncode != 0:
        print(f"  [Build] pnpm install FAILED, trying extra deps...")
        if extra_deps:
            for dep in extra_deps:
                subprocess.run(["pnpm", "add", dep], cwd=str(out_dir), capture_output=True, timeout=120)


# ─── Route Extraction ─────────────────────────────────────────────

def extract_test_routes(gen_dir):
    """Extract routes from App.tsx, handling multi-line and nested <Route> elements.

    Uses indentation to resolve parent-child nesting for relative route paths.
    """
    app_f = gen_dir / "src" / "App.tsx"
    raw = set()
    if app_f.exists():
        content = app_f.read_text(errors="ignore")
        lines = content.split('\n')
        # Collect (indent, path) for each <Route that has a path attribute
        route_entries = []  # (indent_level, path_value)
        path_re = re.compile(r'path\s*=\s*["\']([^"\']*)["\']')
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.lstrip()
            if stripped.startswith('<Route'):
                indent = len(line) - len(stripped)
                # Gather full tag text (may span multiple lines)
                tag_text = stripped
                j = i
                while '>' not in tag_text and j + 1 < len(lines):
                    j += 1
                    tag_text += ' ' + lines[j].strip()
                pm = path_re.search(tag_text)
                if pm:
                    route_entries.append((indent, pm.group(1)))
                i = j + 1
            else:
                i += 1

        # Resolve parent paths using indentation stack
        indent_stack = []  # stack of (indent_level, absolute_path)
        for indent, seg in route_entries:
            # Pop entries with indent >= current (siblings or deeper)
            while indent_stack and indent_stack[-1][0] >= indent:
                indent_stack.pop()
            parent_prefix = indent_stack[-1][1] if indent_stack else ""
            if seg.startswith("/"):
                full = seg
            elif seg:
                full = parent_prefix.rstrip("/") + "/" + seg
            else:
                full = parent_prefix  # index route
            # Clean trailing wildcards
            p_clean = full.rstrip("/*").rstrip("/") if full.endswith("/*") else full
            if p_clean and p_clean != "*":
                norm = "/" + p_clean.strip("/")
                raw.add(norm)
            # Push as potential parent for subsequent deeper-indented routes
            indent_stack.append((indent, full))

    raw.add("/")
    raw = sorted(raw)
    test_routes = []
    for r in raw:
        if ":" in r:
            url = re.sub(r':(\w+)', '1', r)
            test_routes.append({"def": r, "url": url, "type": "dynamic"})
        else:
            test_routes.append({"def": r, "url": r, "type": "concrete"})
    return test_routes


# ─── Error File Extraction ────────────────────────────────────────

def _extract_error_files(error_text, gen_dir):
    """Extract source files referenced in build error output.

    Tries multiple strategies:
      1. Absolute paths:  /path/to/gen/src/Foo.tsx:12:5
      2. Relative paths:  src/Foo.tsx:12:5
      3. 'is not exported' pattern: grep all src/*.tsx for the bad import
      4. Rollup/esbuild plugin context: file mentioned after 'error in' or similar
    Returns list of src-relative paths, or empty list if nothing found.
    """
    gen_dir_str = str(gen_dir)
    found = []

    # Strategy 1: absolute path references
    abs_files = re.findall(rf'{re.escape(gen_dir_str)}/(src/[^\s:,]+\.tsx?)', error_text)
    found.extend(abs_files)

    # Strategy 2: relative path references (src/xxx.tsx:line:col or src/xxx.tsx)
    rel_files = re.findall(r'(?:^|\s)(src/[^\s:,]+\.tsx?)(?::\d+|[\s,])', error_text, re.MULTILINE)
    found.extend(rel_files)

    # Strategy 3: "X" is not exported by "node_modules/..." → grep source for the bad import
    m = re.search(r'"(\w+)" is not exported by "node_modules/([^"]+)"', error_text)
    if m and not found:
        bad_symbol = m.group(1)
        src_dir = gen_dir / "src"
        if src_dir.exists():
            for f in src_dir.rglob("*.tsx"):
                if "node_modules" in str(f):
                    continue
                try:
                    content = f.read_text(errors="ignore")
                    # Check if this file imports the bad symbol
                    if re.search(rf'\b{re.escape(bad_symbol)}\b', content) and \
                       re.search(r'import\s+', content):
                        rel = str(f.relative_to(gen_dir))
                        found.append(rel)
                except:
                    pass
            # Also check .ts files
            for f in src_dir.rglob("*.ts"):
                if "node_modules" in str(f) or f.suffix == ".tsx":
                    continue
                try:
                    content = f.read_text(errors="ignore")
                    if re.search(rf'\b{re.escape(bad_symbol)}\b', content) and \
                       re.search(r'import\s+', content):
                        rel = str(f.relative_to(gen_dir))
                        found.append(rel)
                except:
                    pass

    # Strategy 4: "Transform failed" with file path in esbuild output
    transform_files = re.findall(r'(?:Transform failed|ERROR)\s+.*?(src/[^\s:,]+\.tsx?)', error_text)
    found.extend(transform_files)

    # Deduplicate preserving order
    seen = set()
    result = []
    for f in found:
        if f not in seen:
            seen.add(f)
            result.append(f)

    return result[:3]


# ─── Exec Failure Classification ──────────────────────────────────

def classify_exec_failure(error_text, gen_dir):
    """Classify Exec failure by root cause.

    Returns dict:
      category:    hallucination | inconsistency | integration | syntax
      detected_at: build | runtime
      detail:      human-readable one-liner
    """
    is_runtime = error_text.startswith("RUNTIME_ERROR:")
    detected_at = "runtime" if is_runtime else "build"
    et = error_text.lower()

    # ── hallucination: referencing non-existent external API/export ──
    m = re.search(r'"(\w+)"\s+is\s+not\s+exported\s+by\s+"(node_modules/[^"]+)"', error_text)
    if m:
        return {
            "category": "hallucination",
            "detected_at": detected_at,
            "detail": f'"{m.group(1)}" is not exported by "{m.group(2)}"',
        }
    # Cannot find package (npm package not in package.json)
    m = re.search(r"cannot find (?:package|module)\s+['\"]([^'\"]+)['\"]", et)
    if m and "node_modules" in m.group(0) or (m and not m.group(1).startswith(".")):
        pkg = m.group(1) if m else ""
        if pkg and not pkg.startswith(".") and not pkg.startswith("src"):
            return {
                "category": "hallucination",
                "detected_at": detected_at,
                "detail": f"Cannot find package \"{pkg}\"",
            }

    # ── inconsistency: cross-file mismatch in own generated code ──
    # Export not found in own source files
    m = re.search(r'"(\w+)"\s+is\s+not\s+exported\s+by\s+"(src/[^"]+)"', error_text)
    if m:
        return {
            "category": "inconsistency",
            "detected_at": detected_at,
            "detail": f'"{m.group(1)}" is not exported by "{m.group(2)}"',
        }
    # Cannot resolve own module
    m = re.search(r'(?:could not resolve|cannot find module)\s+["\'](\./[^"\']+|\.\./[^"\']+)["\']', et)
    if m:
        return {
            "category": "inconsistency",
            "detected_at": detected_at,
            "detail": f"Cannot resolve \"{m.group(1)}\"",
        }
    # Runtime: Maximum call stack (component name collision → infinite recursion)
    if "maximum call stack" in et or "too much recursion" in et:
        return {
            "category": "inconsistency",
            "detected_at": detected_at,
            "detail": "Maximum call stack size exceeded (likely component name collision)",
        }

    # ── integration: missing Provider / Router / entry wiring ──
    # Runtime: context/provider errors
    provider_patterns = [
        r"must be used within a (\w+)",
        r"useContext.*undefined",
        r"cannot read properties of (?:undefined|null).*useContext",
        r"no routes matched",
    ]
    for pat in provider_patterns:
        m = re.search(pat, et)
        if m:
            detail = m.group(0)[:100]
            return {
                "category": "integration",
                "detected_at": detected_at,
                "detail": detail,
            }
    # Runtime blank page with no specific console error → likely integration issue
    if is_runtime and ("no route rendered" in et or "browser console errors" not in et):
        return {
            "category": "integration",
            "detected_at": detected_at,
            "detail": "All pages blank (no console error captured, likely missing Provider/Router)",
        }

    # ── syntax: parse / transform / format errors ──
    syntax_patterns = [
        r"parse error",
        r"unexpected token",
        r"transform failed",
        r"unterminated string",
        r"expression expected",
        r"parseast\.js",
    ]
    for pat in syntax_patterns:
        if re.search(pat, et):
            # Extract a short context around the syntax error
            m2 = re.search(r'(\d+)\s*\|(.{0,60})', error_text)
            detail = f"Syntax/parse error"
            if m2:
                detail += f" near line {m2.group(1)}: {m2.group(2).strip()[:60]}"
            return {
                "category": "syntax",
                "detected_at": detected_at,
                "detail": detail,
            }

    # ── fallback ──
    return {
        "category": "syntax" if not is_runtime else "integration",
        "detected_at": detected_at,
        "detail": error_text.splitlines()[0][:120],
    }


# ─── Stage 2: Executability Check ────────────────────────────────────────────

def check_exec(gen_dir, port=5280):
    srv, url = start_server(gen_dir, port)
    if url is None:
        stop_server(srv)
        # Server failed to start — run build to get error message
        build_r = subprocess.run(
            ["npx", "vite", "build"], cwd=str(gen_dir),
            capture_output=True, text=True, timeout=180
        )
        error_text = (build_r.stdout[-2000:] + "\n" + build_r.stderr[-2000:]).strip()
        error_files = _extract_error_files(error_text, gen_dir)
        return False, error_text, error_files
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 900})

            # Capture runtime errors from the browser console
            console_errors = []
            page.on("pageerror", lambda err: console_errors.append(str(err)))

            test_routes = extract_test_routes(gen_dir)
            home_rendered = False
            errors = []
            for tr in test_routes[:15]:
                try:
                    page.goto(f"{url}{tr['url']}", wait_until="domcontentloaded", timeout=10000)
                    page.wait_for_timeout(3500)
                    current_url = page.url
                    if current_url != f"{url}{tr['url']}" and current_url != f"{url}{tr['url']}/":
                        page.wait_for_timeout(1500)
                    root = page.query_selector("#root")
                    if root:
                        visible_text = root.inner_text().strip()
                        # Count descendant elements to distinguish real UI from placeholders
                        el_count = root.evaluate("el => el.querySelectorAll('*').length")
                        # Exec PASS criterion: only the home route '/' counts.
                        # Tracks whether the canonical entry page renders meaningful content.
                        if tr['url'] == '/' and len(visible_text) > 10 and el_count >= 5:
                            home_rendered = True
                except Exception as e:
                    errors.append(str(e)[:100])
            build_r = subprocess.run(
                ["npx", "vite", "build"], cwd=str(gen_dir),
                capture_output=True, text=True, timeout=180
            )
            build_passed = build_r.returncode == 0
            if not build_passed:
                error_text = (build_r.stdout[-2000:] + "\n" + build_r.stderr[-2000:]).strip()
                error_files = _extract_error_files(error_text, gen_dir)
                browser.close()
                if home_rendered:
                    # Dev server's home renders but production build fails (e.g., stricter rollup).
                    # Count as Exec pass since the entry page is functionally correct.
                    return True, error_text, error_files
                return False, error_text, error_files
            browser.close()
            if home_rendered:
                return True, "", []
            # Build passed but home page did not render — runtime error
            runtime_info = "RUNTIME_ERROR: Build passed but home route '/' did not render content.\n"
            if console_errors:
                runtime_info += "Browser console errors:\n" + "\n".join(console_errors[:10]) + "\n"
            runtime_info += f"Home route '/' failed to render meaningful content (>10 chars text + >=5 DOM elements in #root)."
            # Target App.tsx and main.tsx as likely culprits (Provider/wrapper issues)
            runtime_files = []
            for candidate in ["src/App.tsx", "src/main.tsx"]:
                if (gen_dir / candidate).exists():
                    runtime_files.append(candidate)
            print(f"  [Exec] {runtime_info.splitlines()[0]}")
            return False, runtime_info, runtime_files
    finally:
        stop_server(srv)


# ─── Stage 3: Self-Debug Loop ─────────────────────────────────────

def _ask_model_to_locate(client, model, gen_dir, errors):
    """When regex can't find the error file, ask the model to identify it."""
    src_dir = gen_dir / "src"
    file_list = []
    for f in sorted(src_dir.rglob("*.tsx")) + sorted(src_dir.rglob("*.ts")):
        if "node_modules" in str(f):
            continue
        rel = str(f.relative_to(gen_dir))
        file_list.append(rel)

    locate_prompt = make_locate_prompt(errors, file_list)

    response, _ = chat(client, model, [
        {"role": "user", "content": locate_prompt},
    ], max_tokens=256, stage="debug_locate")

    located = []
    for line in response.strip().split("\n"):
        line = line.strip().strip("-").strip()
        if line.startswith("src/") and (line.endswith(".tsx") or line.endswith(".ts")):
            if (gen_dir / line).exists():
                located.append(line)
    return located[:3]


def _apply_fixed_files(model_output, gen_dir, expected_files):
    """Parse multi-file output (--- filepath --- format) and write fixed files."""
    import re as _re
    parts = _re.split(r'^---\s*(src/\S+)\s*---\s*$', model_output, flags=_re.MULTILINE)
    # parts: ['preamble', 'filepath1', 'content1', 'filepath2', 'content2', ...]
    if len(parts) >= 3:
        for i in range(1, len(parts) - 1, 2):
            fpath = parts[i].strip()
            content = parts[i + 1].strip()
            full = gen_dir / fpath
            if full.exists():
                full.write_text(content)
                print(f"    Fixed: {fpath}")
    else:
        # Single file output — apply to first expected file
        if expected_files:
            full = gen_dir / expected_files[0]
            if full.exists():
                full.write_text(model_output.strip())
                print(f"    Fixed: {expected_files[0]}")


def self_debug(client, model, gen_dir, max_k=3, port=5280):
    debug_log = []
    for k in range(1, max_k + 1):
        print(f"  [Debug] Round {k}/{max_k}...")
        exec_pass, errors, error_files = check_exec(gen_dir, port)
        if exec_pass:
            print(f"  [Debug] Exec PASS after round {k-1} (checked at start of round {k})")
            debug_log.append({"round": k, "status": "PASS_CHECK", "note": "Exec already passed"})
            return True, k - 1, debug_log

        is_runtime = errors.startswith("RUNTIME_ERROR:")
        fail_info = classify_exec_failure(errors, gen_dir)

        # If regex extraction found nothing, ask model to locate the file
        if not error_files:
            print(f"  [Debug] No error files extracted by regex, asking model to locate...")
            error_files = _ask_model_to_locate(client, model, gen_dir, errors)
            if error_files:
                print(f"  [Debug] Model identified: {error_files}")
            else:
                print(f"  [Debug] Model could not locate files either, skipping round")
                debug_log.append({"round": k, "status": "NO_FILE",
                                 "fail_info": fail_info, "error_snippet": errors[:200]})
                continue

        print(f"  [Debug] Exec FAIL ({fail_info['category']}/{fail_info['detected_at']}). Files: {error_files}")
        backups = {}
        for fpath in error_files[:3]:
            full_path = gen_dir / fpath
            if full_path.exists():
                backups[fpath] = full_path.read_text(errors="ignore")

        if is_runtime:
            # Runtime error: send all target files together for holistic fix
            file_contents = []
            for fpath in error_files[:3]:
                full_path = gen_dir / fpath
                if full_path.exists():
                    file_contents.append(f'<file path="{fpath}">\n{full_path.read_text(errors="ignore")}\n</file>')
            fix_prompt = make_runtime_fix_prompt(errors, file_contents)
            fixed, _ = chat(client, model, [
                {"role": "user", "content": fix_prompt},
            ], max_tokens=32768, stage=f"debug_k{k}_runtime")
            # Parse multi-file output
            _apply_fixed_files(fixed, gen_dir, error_files[:3])
        else:
            # Build error: fix files individually, include project file listing for context
            src_dir = gen_dir / "src"
            file_listing = "\n".join(
                str(f.relative_to(gen_dir)) for f in sorted(src_dir.rglob("*.tsx")) + sorted(src_dir.rglob("*.ts"))
            ) if src_dir.exists() else ""
            for fpath in error_files[:3]:
                full_path = gen_dir / fpath
                if not full_path.exists():
                    continue
                current_code = full_path.read_text(errors="ignore")
                fix_prompt = make_build_fix_prompt(fpath, errors, file_listing, current_code)
                fixed, _ = chat(client, model, [
                    {"role": "user", "content": fix_prompt},
                ], max_tokens=32768, stage=f"debug_k{k}_{Path(fpath).stem}")
                full_path.write_text(fixed)
                print(f"    Fixed: {fpath}")

        # Verify: for build errors check vite build; for runtime, check_exec in next iteration
        if not is_runtime:
            verify_r = subprocess.run(
                ["npx", "vite", "build"], cwd=str(gen_dir),
                capture_output=True, text=True, timeout=180
            )
            if verify_r.returncode != 0:
                new_error = (verify_r.stdout[-500:] + verify_r.stderr[-500:]).strip()
                old_key_errors = set(re.findall(r'(src/[^\s:]+\.tsx?:\d+)', errors))
                new_key_errors = set(re.findall(r'(src/[^\s:]+\.tsx?:\d+)', new_error))
                if new_key_errors and not (new_key_errors & old_key_errors) and old_key_errors:
                    print(f"    [Debug] Fix introduced new errors, rolling back")
                    for fpath, content in backups.items():
                        (gen_dir / fpath).write_text(content)
                    debug_log.append({"round": k, "status": "ROLLBACK", "files_fixed": error_files[:3],
                                     "fail_info": fail_info, "error_snippet": errors[:200]})
                    continue
        debug_log.append({"round": k, "fail_info": fail_info, "files_fixed": error_files[:3],
                         "error_snippet": errors[:200]})
    exec_pass, _, _ = check_exec(gen_dir, port)
    if exec_pass:
        print(f"  [Debug] Exec PASS after round {max_k}")
        return True, max_k, debug_log
    else:
        print(f"  [Debug] Exec still FAIL after {max_k} rounds")
        return False, max_k, debug_log


# ─── Public API ───────────────────────────────────────────────────

def run_generation(client, model, ss_dir, run_dir, gen_dir, port=5280, skip_gen=False):
    """Run Stage 1-3. Returns (exec_at1, exec_at3, debug_rounds, debug_log).

    If skip_gen=False, generates code into gen_dir and saves plan/raw_output.
    Always runs Exec check and self-debug.
    """
    # ── Stage 1: Generate ──
    if not skip_gen:
        print(f"\n--- Stage 1: Code Generation ---")
        generated, extra_deps, plan_data = generate_code(client, model, ss_dir, run_dir=run_dir)
        write_project(generated, extra_deps, gen_dir)
        (run_dir / "plan.json").write_text(json.dumps(plan_data, indent=2))
        (run_dir / "raw_output.json").write_text(json.dumps(generated, indent=2, ensure_ascii=False))
    else:
        print(f"\n--- Stage 1: SKIPPED (using existing generated code) ---")

    # ── Stage 2: Exec@1 ──
    print(f"\n--- Stage 2: Executability Check (@1) ---")
    exec_at1, errors_at1, _ = check_exec(gen_dir, port)
    exec_fail_info = None
    if not exec_at1:
        exec_fail_info = classify_exec_failure(errors_at1, gen_dir)
        print(f"  Exec@1 = FAIL  category={exec_fail_info['category']}  detected_at={exec_fail_info['detected_at']}")
        print(f"         detail: {exec_fail_info['detail'][:120]}")
    else:
        print(f"  Exec@1 = PASS")

    # ── Stage 3: Self-Debug → Exec@3 ──
    exec_at3 = exec_at1
    debug_rounds = 0
    debug_log = []
    if not exec_at1:
        print(f"\n--- Stage 3: Self-Debug Loop ---")
        exec_at3, debug_rounds, debug_log = self_debug(client, model, gen_dir, max_k=3, port=port)
        print(f"  Exec@3 = {'PASS' if exec_at3 else 'FAIL'} (rounds={debug_rounds})")
    else:
        print(f"\n--- Stage 3: Self-Debug SKIPPED (Exec@1 passed) ---")

    return exec_at1, exec_at3, debug_rounds, debug_log, exec_fail_info
