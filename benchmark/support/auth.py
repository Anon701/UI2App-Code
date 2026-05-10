"""
UI2App auth + gate handling.

Three source-derived mitigations applied to the generated dev server
(no per-app human configuration), described in paper Appendix VFS:
  (i) localStorage auth seeding
 (ii) in-memory auth form-fill
(iii) splash/intro gate dismissal
"""

import json, os, re, subprocess, sys, hashlib, time
from pathlib import Path

import numpy as np


# ─── localStorage auth seeding ─────────────────────────────────────

# Permissive auth payload — covers the common keys models use across many
# auth implementations. The shape is a union of fields seen in the wild
# (e.g., kimi-k2.5 reads `{isAuthenticated, user}`, others read `{loggedIn,
# token}`, others `{accessToken}`). Any reasonable check will see "true".
_AUTH_PAYLOAD_TEMPLATE = {
    "isAuthenticated": True,
    "isAuth": True,
    "loggedIn": True,
    "authenticated": True,
    "user": {"name": "Eval User", "email": "<anon@example.com>", "id": "eval-1",
             "username": "evaluser", "role": "admin", "avatar": ""},
    "token": "eval-token-12345",
    "accessToken": "eval-token-12345",
    "id": "eval-1",
    "email": "<anon@example.com>",
}
_AUTH_KEY_RE = re.compile(r"(auth|session|login|signin|loggedin|token|user)", re.I)


def detect_localstorage_seeds(project_dir):
    """Scan src/**/*.[jt]s(x) for `localStorage.getItem('xxx')` and
    `localStorage.setItem('xxx', ...)` patterns whose key looks auth-related,
    and return a list of `(key, json_value)` pairs to inject before
    navigation. The injection makes auth gates (e.g., useAuth hooks that
    initialise from localStorage) start in the authenticated state, allowing
    the pipeline to render real route content rather than the login form.

    Detection is intentionally generous: any localStorage key matching
    /(auth|session|login|signin|loggedin|token|user)/i becomes a seed slot.
    The injected payload is a permissive union of common auth-state shapes.
    """
    project_dir = Path(project_dir)
    src_root = project_dir / "src"
    if not src_root.is_dir():
        return []
    seen_keys = set()
    seeds = []
    payload = json.dumps(_AUTH_PAYLOAD_TEMPLATE)
    for f in src_root.rglob("*.[jt]s*"):
        try:
            src = f.read_text(errors="ignore")
        except Exception:
            continue
        for m in re.finditer(r"""localStorage\.(?:getItem|setItem)\s*\(\s*['"]([^'"]+)['"]""", src):
            key = m.group(1)
            if key in seen_keys:
                continue
            if _AUTH_KEY_RE.search(key):
                seen_keys.add(key)
                seeds.append((key, payload))
    return seeds


def _apply_localstorage_seeds(context, seeds):
    """Register init scripts on a Playwright context so every page navigated
    through it has the seeds set before any app JS runs."""
    if not seeds:
        return
    parts = []
    for key, value in seeds:
        parts.append(f"window.localStorage.setItem({json.dumps(key)}, {json.dumps(value)});")
    context.add_init_script("try {\n" + "\n".join(parts) + "\n} catch(_) {}")


def detect_inmemory_auth_pattern(gen_dir):
    """Detect React Context auth state that lives in memory (no localStorage,
    no sessionStorage) — typical Sonnet 4.6 pattern: AuthContext.tsx with
    `const [isAuthenticated, setIsAuthenticated] = useState(false)` and an
    `<ProtectedRoute>` wrapper that redirects unauthenticated to /login.

    Returns True if found. When True, localStorage seeding is insufficient
    and the pipeline must fall back to form-fill login + SPA navigation to
    bypass the gate (cookies/session won't help either since state is purely
    React-internal).
    """
    gen_dir = Path(gen_dir)
    src = gen_dir / "src"
    if not src.is_dir():
        return False
    auth_id_re = re.compile(r"\b(isAuthenticated|isLoggedIn|isLogin|loggedIn|currentUser|authUser)\b", re.I)
    state_false_re = re.compile(r"useState\s*(?:<[^>]+>)?\s*\(\s*false\s*\)")
    # Heuristic: look at *Context*.[jt]sx and *useAuth*.[jt]s* — auth state lives there
    candidates = list(src.rglob("*[Cc]ontext*.[jt]sx")) + list(src.rglob("*[Uu]se[Aa]uth*.[jt]s*"))
    for f in candidates:
        try:
            txt = f.read_text(errors="ignore")
        except Exception:
            continue
        # Must have useState(false) AND auth identifier AND no localStorage hookup
        if state_false_re.search(txt) and auth_id_re.search(txt):
            if "localStorage" not in txt and "sessionStorage" not in txt:
                return True
    return False


def _attempt_form_login(page, app_url, gen_routes, rendered):
    """Locate the gen app's /login route, fill its form, submit, and return
    True if login appears to have succeeded. Used as a recovery for in-memory
    React auth where localStorage seeding can't bypass the gate.

    The function looks for a route whose path contains "login" or "signin"
    (case-insensitive). On that page it fills the first email-like and the
    first password input, clicks the most likely submit button, and treats
    success as "URL changed away from /login OR DOM no longer contains
    visible password input".
    """
    login_route = None
    for tr in gen_routes:
        path = tr["def"].lower()
        if any(tok in path for tok in ("/login", "/signin", "/sign-in", "/auth/login")):
            login_route = tr
            break
    if login_route is None:
        return False
    try:
        page.goto(f"{app_url}{login_route['url']}", wait_until="domcontentloaded", timeout=12000)
        page.wait_for_timeout(2000)
        email = page.locator(
            'input[type="email"], input[name="email" i], input[id*="email" i], '
            'input[name="username" i], input[autocomplete="email"]'
        ).first
        pwd = page.locator('input[type="password"], input[name="password" i]').first
        if email.count() == 0 or pwd.count() == 0:
            return False
        email.fill("<anon@example.com>")
        pwd.fill("EvalPassword!23")
        # Try submit button candidates in order of specificity
        for sel in ['button[type="submit"]',
                    'button:has-text("sign in")', 'button:has-text("login")',
                    'button:has-text("log in")', 'button:has-text("submit")',
                    'button:has-text("continue")', 'form button']:
            try:
                btn = page.locator(sel).first
                if btn.count() > 0:
                    btn.click(timeout=2500)
                    break
            except Exception:
                continue
        page.wait_for_timeout(3000)
        # Heuristic success: URL changed OR no password field on page now
        cur_url = page.url
        on_login = login_route["url"].rstrip("/") in cur_url.rstrip("/") and len(cur_url.split("?")[0].rstrip("/")) <= len(f"{app_url}{login_route['url']}".rstrip("/"))
        pwd_still_visible = False
        try:
            pwd_now = page.locator('input[type="password"]').first
            pwd_still_visible = pwd_now.count() > 0 and pwd_now.is_visible()
        except Exception:
            pass
        return (not on_login) or (not pwd_still_visible)
    except Exception:
        return False



def _spa_goto(page, route_path):
    """Navigate within a single page session via History API push + popstate
    so React Router re-renders without a full reload (which would wipe
    in-memory React state, including the just-set auth flag)."""
    try:
        page.evaluate(
            "(p) => { window.history.pushState({}, '', p); "
            "window.dispatchEvent(new PopStateEvent('popstate')); }",
            route_path,
        )
        return True
    except Exception:
        return False



# ─── Gate detection (intro/splash/auth screens that block routing) ─

# Common dismiss-button label tokens (case-insensitive substring match) used
# both as a fallback when AST extraction fails and as the runtime click set.
_GENERIC_DISMISS_TOKENS = [
    "skip intro", "skip", "continue", "get started", "let's go",
    "enter", "enter app", "launch", "begin", "start", "i'm ready",
    "got it", "accept", "agree", "ok", "dismiss", "close",
]

def _generic_dismiss_selectors():
    sels = []
    for t in _GENERIC_DISMISS_TOKENS:
        sels.append(f'button:has-text("{t}")')
        sels.append(f'[role="button"]:has-text("{t}")')
        sels.append(f'a:has-text("{t}")')
    return sels


def _scan_gate_component(comp_path: Path):
    """Open the gate component file and try to extract the dismiss-button label.

    Looks for `<button onClick={onXxx}> Label </button>` where onXxx names a
    callback the parent passes (onDone/onComplete/onContinue/...). Falls back to
    the first button label in the file if no callback-bound button is found.
    Returns a list of CSS selectors (most specific first).
    """
    if not comp_path.exists():
        return []
    src = comp_path.read_text(errors="ignore")
    # Strict pass: button bound to a known callback prop
    btn = re.search(
        r'<button[^>]*onClick\s*=\s*\{(?:on(?:Done|Complete|Continue|Finish|Start|Enter|Skip|Dismiss|Ready|Begin|Login|Auth|Accept))\b[^}]*\}[^>]*>(.*?)</button>',
        src, re.DOTALL,
    )
    if not btn:
        # Loose pass: any button onClick
        btn = re.search(r'<button[^>]*onClick[^>]*>(.*?)</button>', src, re.DOTALL)
    if not btn:
        return []
    # Strip nested JSX (e.g., trailing icon) — keep only plain text fragments.
    inner = re.sub(r'<[^>]+>', ' ', btn.group(1))
    inner = re.sub(r'\{[^}]+\}', ' ', inner)
    label = re.sub(r'\s+', ' ', inner).strip()
    if not label or len(label) > 50:
        return []
    # Quote the label so it survives the Playwright selector parser.
    safe = label.replace('"', '\\"')
    return [f'button:has-text("{safe}")', f'[role="button"]:has-text("{safe}")']


def detect_gates(gen_dir: Path):
    """Scan src/App.tsx for patterns that prevent the Router from rendering.

    Recognised patterns (most to least common in the curated dataset):
      A. useState(false) + early `if (!flag) return <SomeComponent .../>` *before*
         the BrowserRouter / Routes block. Phineas-portfolio's IntroScreen is the
         canonical example.
      B. `<Navigate to="/...">` redirects on `/`. Not a true gate but produces
         shadow eval_screenshots; reported but no dismissal is needed.

    Returns:
        {
            "has_gate": bool,                # True only for pattern A
            "pattern": "useState_early_return" | "navigate_redirect" | None,
            "gate_component": str | None,
            "dismiss_selectors": list[str],  # ordered: AST-extracted first, generic fallback after
            "redirects": list[(from, to)],   # for diagnostics only
        }
    """
    out = {"has_gate": False, "pattern": None, "gate_component": None,
           "dismiss_selectors": [], "redirects": []}
    app = gen_dir / "src" / "App.tsx"
    if not app.exists():
        return out
    src = app.read_text(errors="ignore")

    # Pattern B: collect <Navigate> redirects (informational, not gating).
    for m in re.finditer(r'<Navigate\s+[^>]*to=\{?\s*["\']([^"\']+)["\']', src):
        out["redirects"].append(("/", m.group(1)))

    # Pattern A: useState(false) + early return of a component before <BrowserRouter>/<Routes>.
    flag_match = re.search(r'useState\s*(?:<[^>]+>)?\s*\(\s*(?:false|true)\s*\)', src)
    if not flag_match:
        return out
    # Find an early `if (...) return <X ...` that fires before BrowserRouter/Routes.
    router_idx = re.search(r'<(?:BrowserRouter|HashRouter|Routes)\b', src)
    router_pos = router_idx.start() if router_idx else len(src)
    early = re.search(r'if\s*\([^)]+\)\s*\{?\s*return\s*\(?\s*<(\w+)\b', src[:router_pos])
    if not early:
        return out
    comp_name = early.group(1)
    # Skip Fragment / lowercase HTML tags / Router primitives.
    if comp_name[:1].islower() or comp_name in {"Fragment"}:
        return out

    out["has_gate"] = True
    out["pattern"] = "useState_early_return"
    out["gate_component"] = comp_name

    # Resolve the gate component's source file via its import statement.
    imp = re.search(rf'import\s+\w+\s+from\s+["\']([^"\']+)["\']\s*\n?\s*(?=.*<{comp_name}\b)', src, re.DOTALL)
    imp = re.search(rf'import\s+\w+\s+from\s+["\'](\.[^"\']+)["\']', src) if not imp else imp
    # More targeted: find the line that imports comp_name specifically.
    for line in src.splitlines():
        if f' {comp_name} ' in f' {line} ' or f'{{{comp_name}}}' in line.replace(' ', ''):
            mm = re.search(r'from\s+["\']([^"\']+)["\']', line)
            if mm:
                imp = mm; break
    if isinstance(imp, re.Match) and imp.group(1).startswith('.'):
        rel = imp.group(1)
        for ext in (".tsx", ".ts", ".jsx", ".js", "/index.tsx", "/index.ts"):
            cand = (app.parent / (rel + ext)).resolve()
            if cand.exists():
                out["dismiss_selectors"] = _scan_gate_component(cand)
                break

    # Always append generic fallbacks after AST-derived ones.
    out["dismiss_selectors"] += _generic_dismiss_selectors()
    return out


def _try_dismiss_gate(page, selectors, per_attempt_timeout=400):
    """Try each selector in order; return on first successful click.

    Returns the matched selector on success, or None if every candidate
    failed. When None is returned, downstream DOM extraction will be on
    the gate state instead of the routed content -- caller should be
    aware that VFS for that route may be artificially low.
    """
    for sel in selectors:
        try:
            loc = page.locator(sel).first
            if loc.count() == 0:
                continue
            loc.click(timeout=per_attempt_timeout)
            return sel
        except Exception:
            continue
    if selectors:
        print(f"    [warn] gate dismissal failed: tried {len(selectors)} selectors, none clickable")
    return None

