#!/usr/bin/env python3
"""
UI2App project discovery, paths, and React scaffold.

Layout (release):
    UI2App-Code/
    ├── data/
    │   ├── apps/<app_id>/{manifest.json, img_*.png}   ← screenshots from HF
    │   ├── sources/<app_id>/                          ← cloned repos for VFS DOM mode
    │   └── runs/<app_id>/<model>/run_<ts>/            ← pipeline output
    └── benchmark/                                      ← code only
"""

import json
from pathlib import Path
from dotenv import load_dotenv

# ─── Paths ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent  # UI2App-Code
load_dotenv(ROOT / ".env", override=False)

DATA_DIR     = ROOT / "data"
APPS_DIR     = DATA_DIR / "apps"      # screenshots + manifest from HF
SOURCES_DIR  = DATA_DIR / "sources"   # cloned source repos for VFS DOM
RUNS_DIR     = DATA_DIR / "runs"      # pipeline output
COST_LOG     = RUNS_DIR / "cost_log" / "api_costs.jsonl"

ENTRY_FILES = {"App.tsx", "App.ts", "main.tsx", "main.ts", "index.tsx", "index.ts"}


def discover_projects(project_filter=None):
    """Auto-discover apps from data/apps/<app_id>/ directories.

    Each <app_id>/ must contain manifest.json (HF dataset format).

    Args:
        project_filter: list of app_ids to include (None = all)
    """
    projects = {}
    if not APPS_DIR.exists():
        return projects
    for d in sorted(APPS_DIR.iterdir()):
        if d.is_dir() and (d / "manifest.json").exists():
            projects[d.name] = {
                "screenshot_dir": d,           # data/apps/<id>/ holds both manifest + PNGs
                "project_dir":    d,           # alias retained for downstream code
                "source_dir":     SOURCES_DIR / d.name,  # convention path
                "run_root":       RUNS_DIR / d.name,     # output goes here
            }
    if project_filter:
        projects = {k: v for k, v in projects.items() if k in project_filter}
    return projects


PROJECTS = discover_projects()


# ─── Scaffold ─────────────────────────────────────────────────────
SCAFFOLD = {
    "package.json": json.dumps({
        "name": "generated-app", "private": True, "version": "0.0.0", "type": "module",
        "scripts": {"dev": "vite", "build": "vite build", "preview": "vite preview"},
        "dependencies": {"react": "^18.3.1", "react-dom": "^18.3.1",
                         "react-router-dom": "^6.28.0", "lucide-react": "^0.460.0"},
        "devDependencies": {"@types/react": "^18.3.12", "@types/react-dom": "^18.3.1",
                            "@vitejs/plugin-react": "^4.3.4", "autoprefixer": "^10.4.20",
                            "postcss": "^8.4.49", "tailwindcss": "^3.4.15",
                            "typescript": "~5.6.2", "vite": "^6.0.0"},
    }, indent=2),
    "vite.config.ts": "import { defineConfig } from 'vite'\nimport react from '@vitejs/plugin-react'\nexport default defineConfig({ plugins: [react()], server: { host: true } })",
    "tsconfig.json": json.dumps({
        "compilerOptions": {"target": "ES2020", "useDefineForClassFields": True,
            "lib": ["ES2020", "DOM", "DOM.Iterable"], "module": "ESNext",
            "skipLibCheck": True, "moduleResolution": "bundler",
            "allowImportingTsExtensions": True, "isolatedModules": True,
            "moduleDetection": "force", "noEmit": True, "jsx": "react-jsx",
            "strict": True, "noUnusedLocals": False, "noUnusedParameters": False,
            "allowJs": True, "esModuleInterop": True,
            "forceConsistentCasingInFileNames": True, "resolveJsonModule": True},
        "include": ["src"],
    }, indent=2),
    "tailwind.config.js": '/** @type {import("tailwindcss").Config} */\nexport default { content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"], theme: { extend: {} }, plugins: [] }',
    "postcss.config.js": "export default { plugins: { tailwindcss: {}, autoprefixer: {} } }",
    "index.html": '<!doctype html>\n<html lang="en">\n<head><meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1.0"/><title>App</title></head>\n<body><div id="root"></div><script type="module" src="/src/main.tsx"></script></body>\n</html>',
    "src/main.tsx": 'import { StrictMode } from "react"\nimport { createRoot } from "react-dom/client"\nimport App from "./App"\nimport "./index.css"\ncreateRoot(document.getElementById("root")!).render(<StrictMode><App /></StrictMode>)',
    "src/index.css": "@tailwind base;\n@tailwind components;\n@tailwind utilities;\n",
}
