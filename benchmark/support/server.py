#!/usr/bin/env python3
"""
UI2App dev server start/stop + port management.
"""

import json, os, signal, subprocess, time, sys
from pathlib import Path

def _port_free(port):
    """True if nothing is bound to the given TCP port on loopback."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def _wait_port_free(port, timeout=8.0):
    """Block until the port is free, or return False after timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _port_free(port):
            return True
        time.sleep(0.2)
    return False


def start_server(proj_dir, port):
    """Start dev server. Auto-detects framework: next > react-scripts > vite > webpack > vue-cli."""
    pkg_path = proj_dir / "package.json"
    # Auto-install node_modules if missing — covers --skip-gen on older runs where
    # node_modules was cleaned up, as well as any other path that starts the server
    # without an explicit install step.
    if pkg_path.exists() and not (proj_dir / "node_modules").exists():
        print(f"  [Server] node_modules missing, running pnpm install...")
        r = subprocess.run(["pnpm", "install"], cwd=str(proj_dir),
                           capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            print(f"  [Server] pnpm install FAILED: {r.stderr[-400:]}")
            return None, None
    cmd = ["npx", "vite", "--port", str(port), "--strictPort"]
    max_wait = 20
    env = os.environ.copy()
    if pkg_path.exists():
        pkg = json.loads(pkg_path.read_text())
        deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
        scripts = pkg.get("scripts", {})
        if "@umijs/max" in deps or "umi" in deps:
            cmd = ["npx", "max", "dev"]
            env["PORT"] = str(port)
            max_wait = 60
        elif "next" in deps:
            cmd = ["npx", "next", "dev", "-p", str(port)]
            max_wait = 35
        elif "astro" in deps:
            cmd = ["npx", "astro", "dev", "--port", str(port)]
            max_wait = 30
        elif "@vue/cli-service" in deps:
            # Vue CLI (vue-cli-service serve)
            cmd = ["npx", "vue-cli-service", "serve", "--port", str(port)]
            env["NODE_OPTIONS"] = "--openssl-legacy-provider"
            max_wait = 60
        elif "react-scripts" in deps or "react-scripts start" in scripts.get("start", ""):
            cmd = ["npx", "react-scripts", "start"]
            env["PORT"] = str(port)
            env["BROWSER"] = "none"
            # Older CRA / OpenSSL 3.x compat: set --openssl-legacy-provider
            env["NODE_OPTIONS"] = env.get("NODE_OPTIONS", "") + " --openssl-legacy-provider"
            max_wait = 40
        elif "webpack-dev-server" in deps:
            # Ejected CRA with custom scripts/start.js
            if (proj_dir / "scripts" / "start.js").exists():
                cmd = ["node", "scripts/start.js"]
                env["PORT"] = str(port)
                env["BROWSER"] = "none"
                env["NODE_OPTIONS"] = env.get("NODE_OPTIONS", "") + " --openssl-legacy-provider"
                max_wait = 40
            else:
                # Detect custom webpack config in build/ or config/
                wp_cfg = None
                for candidate in ["build/webpack.config.client.dev.js",
                                  "config/webpack.config.dev.js",
                                  "webpack.config.dev.js",
                                  "webpack.config.js"]:
                    if (proj_dir / candidate).exists():
                        wp_cfg = candidate
                        break
                if wp_cfg:
                    cmd = ["npx", "webpack-dev-server", "--config", wp_cfg,
                           "--port", str(port)]
                else:
                    cmd = ["npx", "webpack-dev-server", "--port", str(port)]
                max_wait = 30

    # Wait for the port to be free before --strictPort binds, to avoid
    # a TIME_WAIT / leftover-process race from the previous debug round.
    if not _wait_port_free(port):
        print(f"  [Server] Port {port} still busy after 8s; attempting start anyway")

    srv = subprocess.Popen(
        cmd, cwd=str(proj_dir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=env, preexec_fn=os.setsid,
    )
    # urllib.urlopen respects HTTP_PROXY; force direct for localhost probe
    # so a system proxy (Clash etc.) can't interpose on the readiness check.
    import urllib.request, urllib.error
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    url = f"http://localhost:{port}"
    req = urllib.request.Request(url, headers={"Accept": "*/*"})
    for _ in range(max_wait):
        time.sleep(1)
        if srv.poll() is not None:
            print(f"  [Server] Process exited with code {srv.returncode} on port {port}")
            return srv, None
        try:
            opener.open(req, timeout=3)
            return srv, url
        except urllib.error.HTTPError:
            # Server is up but returned non-200 (e.g. 404 for SPA) — still counts as ready
            return srv, url
        except Exception:
            # Connection refused / server not yet ready -- retry until timeout
            pass
    print(f"  [Server] Timeout waiting for server on port {port}")
    return srv, None


def stop_server(srv):
    if not srv:
        return
    try:
        os.killpg(os.getpgid(srv.pid), signal.SIGTERM)
    except Exception:
        try:
            srv.kill()
        except Exception:
            pass
    # Wait for the process group to actually exit; SIGKILL if it doesn't.
    try:
        srv.wait(timeout=5)
    except Exception:
        try:
            os.killpg(os.getpgid(srv.pid), signal.SIGKILL)
        except Exception:
            try:
                srv.kill()
            except Exception:
                pass
        try:
            srv.wait(timeout=2)
        except Exception:
            print(f"  [warn] dev server pid={srv.pid} did not exit after SIGKILL; "
                  f"its port may remain busy until the OS reclaims it")
