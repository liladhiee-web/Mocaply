<h1 align="center">Mocaply</h1>

<p align="center">
  <b>Load any video clip. AI extracts the full 3D motion, fingers included, and any rigged 3D character performs it in real time.</b><br>
  A video from your camera, phone, or internet extracts high-precision 3D human motion via GPU-accelerated GVHMR + WiLoR. Enjoy lag-free 60 FPS in-browser rendering with zero CPU bottleneck. Export directly to GLB for Blender or game engines.
</p>

<p align="center">
  <a href="https://github.com/liladhiee-web/Mocaply"><img alt="maintained by: liladhiee-web" src="https://img.shields.io/badge/maintainer-liladhiee--web-blue?style=flat-square"></a>
  <a href="https://colab.research.google.com/github/liladhiee-web/Mocaply/blob/main/mocaply_colab.ipynb"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
  <a href="#license"><img alt="license: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-green?style=flat-square"></a>
</p>

---

## How It Works

```mermaid
flowchart LR
  C["Any Video Clip<br/>(mp4 · webm · mov)"] --> H["GVHMR · GPU (Colab or Local)<br/>3D body trajectory & rotation"]
  C --> F["WiLoR · GPU<br/>wrist & finger tracking"]
  H --> R[Mixamo Rig]
  F --> R
  R --> P["Lag-free 60 FPS 3D Playback"]
  R --> G[".glb Export · Blender render"]
```

| Stage | Where | Time |
|---|---|---|
| **HQ** 3D motion: GVHMR → SMPL rotations → Mixamo rig | GPU worker (Free Google Colab or Local) | ~3.5–4s (per 5s clip) |
| **Fingers**: WiLoR / MANO hand tracking | GPU worker | +~0.3s |
| **Playback**: Synchronized video + 3D rigged character | Browser (WebGL Three.js) | Lag-free 60 FPS (<0.1ms/frame) |

---

## Quick Start

### Option A: Run on Google Colab (Recommended — Free NVIDIA T4 GPU)

No local gaming/NVIDIA GPU required. Run both the GVHMR worker and the web interface in Google Colab, exposed over a free Cloudflare tunnel:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liladhiee-web/Mocaply/blob/main/mocaply_colab.ipynb)

1. Open the [Mocaply Colab Notebook](mocaply_colab.ipynb).
2. Set Runtime to **T4 GPU** (**Runtime → Change runtime type → T4 GPU → Save**).
3. Run the setup and launch cells.
4. Click the generated **Cloudflare Tunnel URL** (`https://...trycloudflare.com`) to open Mocaply on your computer or phone!

---

### Option B: Run Locally

**Requirements:** Node.js 18+. For the HQ GPU worker: Windows or Linux, an NVIDIA GPU, Python 3.10+, and `ffmpeg` on PATH.

```bash
git clone https://github.com/liladhiee-web/Mocaply.git
cd Mocaply
node server.mjs               # → http://localhost:5177 (the live preview works immediately)
```

**HQ GPU Worker (Optional, for GVHMR 3D trajectory):**

* **Windows:**
  ```powershell
  hq\setup.ps1
  hq\.venv\Scripts\python hq\worker.py
  ```

* **Linux / WSL:**
  ```bash
  bash hq/setup.sh
  python3 hq/worker.py
  ```

---

## Features

- **Drag & Drop Upload:** Load any local `.mp4`, `.webm`, or `.mov` clip directly from your desktop.
- **Mixamo Rigging:** Built-in support for X Bot, Michelle, and Soldier, plus custom `.fbx` or `.glb` Mixamo character upload.
- **Finger Articulation:** WiLoR MANO hands or MediaPipe Hands fallback captures realistic finger poses and hand gestures.
- **Root Motion Recovery:** Automatic gravity-aware hip trajectory tracking preserves spatial movement.
- **GLB Export:** Download skinned characters with animated motion tracks ready for Blender, Unreal Engine, or Unity.

---

## Project Structure

| Path | Description |
|---|---|
| `server.mjs` | Zero-dependency Node.js HTTP server; serves static files, upload API, and reverse proxies the worker |
| `public/` | Frontend web application (Three.js stage, MediaPipe vision pipeline, retargeting engine) |
| `hq/worker.py` | FastAPI worker providing GVHMR SMPL 3D pose extraction |
| `hq/hands.py` | Full-resolution finger and hand tracking (WiLoR / MediaPipe) |
| `tools/colab_runner.py` | Automated Colab launcher with zero-config Cloudflare Tunneling |
| `mocaply_colab.ipynb` | Google Colab interactive notebook |

---

## Author & Maintainer

**Mocaply** is maintained and updated by [**liladhiee-web**](https://github.com/liladhiee-web).

---

## License & Acknowledgements

Mocaply's code is licensed under the [Apache-2.0 License](LICENSE).

This project integrates and builds upon foundational computer vision research:
- [GVHMR](https://github.com/zju3dv/GVHMR) (ZJU3DV)
- [SMPL-X](https://smpl-x.is.tue.mpg.de/) (Max Planck Institute for Intelligent Systems)
- [WiLoR](https://github.com/rolpotamias/WiLoR)
- [Motion2MixamoRig](https://github.com/MaxLiu22/Motion2MixamoRig)
- [MediaPipe](https://ai.google.dev/edge/mediapipe) & [Three.js](https://threejs.org/)
