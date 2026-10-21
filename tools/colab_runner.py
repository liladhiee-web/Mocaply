#!/usr/bin/env python3
"""Mocaply Google Colab & Remote Runner with Cloudflare Tunnel.

This script:
1. Verifies GPU / CUDA and installs prerequisites if missing.
2. Starts the GVHMR HQ worker (port 5178).
3. Starts the Node.js web server (port 5177).
4. Automatically provisions a free Cloudflare tunnel and outputs the public URL.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HQ_DIR = ROOT / "hq"


def log(msg: str) -> None:
    print(f"\033[1;36m[mocaply]\033[0m {msg}", flush=True)


def download_file(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"Downloading {dest.name}...")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, str(tmp))
    tmp.replace(dest)


def ensure_cloudflared() -> str:
    """Ensure cloudflared binary is available."""
    system = platform.system().lower()
    bin_name = "cloudflared.exe" if system == "windows" else "cloudflared"
    existing = shutil.which(bin_name) or shutil.which("cloudflared")
    if existing:
        return existing

    colab_bin = ROOT / "tools" / bin_name
    if colab_bin.exists():
        if system != "windows":
            colab_bin.chmod(0o755)
        return str(colab_bin)

    log("Downloading Cloudflare tunnel client (cloudflared)...")
    if system == "linux":
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    elif system == "windows":
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
    elif system == "darwin":
        arch = "arm64" if platform.machine() == "arm64" else "amd64"
        url = f"https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-{arch}"
    else:
        raise RuntimeError(f"Unsupported OS: {system}")

    download_file(url, colab_bin)
    if system != "windows":
        colab_bin.chmod(0o755)
    return str(colab_bin)


def ensure_models() -> None:
    """Ensure SMPLX and MediaPipe models are present."""
    smplx_dir = HQ_DIR / "body_models" / "smplx"
    smplx_dir.mkdir(parents=True, exist_ok=True)
    smplx_file = smplx_dir / "SMPLX_NEUTRAL.npz"
    if not smplx_file.exists():
        log("SMPLX_NEUTRAL.npz not found. Downloading from mirror...")
        download_file(
            "https://huggingface.co/camenduru/SMPLer-X/resolve/main/SMPLX_NEUTRAL.npz",
            smplx_file,
        )

    task_file = HQ_DIR / "weights" / "hand_landmarker.task"
    if not task_file.exists():
        log("Downloading MediaPipe hand tracking model...")
        download_file(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
            task_file,
        )


def main() -> int:
    os.chdir(ROOT)
    log("Starting Prompt2Rig Colab Runner...")

    # Check CUDA
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            log(f"Found GPU: {gpu_name}")
        else:
            log("\033[1;33mWarning: CUDA not detected! GVHMR requires a GPU.\033[0m")
    except ImportError:
        pass

    # Check Node.js
    if not shutil.which("node"):
        log("\033[1;31mError: Node.js is required but not found in PATH.\033[0m")
        return 1

    ensure_models()
    cf_path = ensure_cloudflared()

    procs: list[subprocess.Popen] = []

    def cleanup(*_):
        log("Shutting down processes...")
        for p in procs:
            try:
                p.terminate()
                p.kill()
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # 1. Start Python GVHMR worker
    log("Starting GVHMR HQ worker on 127.0.0.1:5178...")
    env = os.environ.copy()
    env["GVHMR_CHECKPOINTS"] = str(HQ_DIR / "weights")
    env["GVHMR_BODY_MODELS"] = str(HQ_DIR / "body_models")
    worker_proc = subprocess.Popen(
        [sys.executable, str(HQ_DIR / "worker.py")],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    procs.append(worker_proc)

    # 2. Start Node.js Web Server
    log("Starting Web Server on port 5177...")
    node_proc = subprocess.Popen(
        ["node", "server.mjs"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    procs.append(node_proc)

    # Give server a moment to start
    time.sleep(1.5)

    # 3. Start Cloudflare Tunnel
    log("Starting Cloudflare Tunnel on http://127.0.0.1:5177...")
    tunnel_proc = subprocess.Popen(
        [cf_path, "tunnel", "--url", "http://127.0.0.1:5177"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    procs.append(tunnel_proc)

    public_url = None
    url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")

    # Monitor output for public tunnel URL
    start_wait = time.time()
    while time.time() - start_wait < 30 and not public_url:
        line = tunnel_proc.stdout.readline() if tunnel_proc.stdout else ""
        if line:
            m = url_pattern.search(line)
            if m:
                public_url = m.group(0)
                break

    if public_url:
        print("\n" + "=" * 65)
        print(f"\033[1;32m  MOCAPLY IS RUNNING ON CLOUDFLARE TUNNEL!\033[0m")
        print(f"  Public URL: \033[1;34m\033[4m{public_url}\033[0m")
        print("=" * 65 + "\n", flush=True)

        # In Colab/IPython, render HTML button
        try:
            from IPython.display import HTML, display

            display(
                HTML(
                    f"""
                <div style="padding:16px;background:#1a1921;border-radius:8px;border:1px solid #302b36;color:#f2ece2;font-family:sans-serif;">
                    <h3 style="margin-top:0;color:#5ab8ff;">Mocaply is Ready!</h3>
                    <p>Click below to open the app with GVHMR GPU acceleration active:</p>
                    <a href="{public_url}" target="_blank" style="display:inline-block;padding:10px 20px;background:#ff3cd2;color:#ffffff;text-decoration:none;border-radius:4px;font-weight:bold;">
                        Open Mocaply Web App
                    </a>
                    <p style="margin-bottom:0;margin-top:10px;font-size:12px;color:#8e8796;">Tunnel URL: <a href="{public_url}" target="_blank" style="color:#ffc94a;">{public_url}</a></p>
                </div>
            """
                )
            )
        except Exception:
            pass
    else:
        log("Warning: Could not detect Cloudflare Tunnel URL automatically.")
        log("Check tunnel output below:")

    # Keep running and streaming logs
    try:
        while True:
            time.sleep(1)
            # If any process died unexpectedly, log it
            for name, proc in [("Worker", worker_proc), ("Server", node_proc), ("Tunnel", tunnel_proc)]:
                ret = proc.poll()
                if ret is not None:
                    log(f"\033[1;31m{name} exited with code {ret}!\033[0m")
                    cleanup()
    except KeyboardInterrupt:
        cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
