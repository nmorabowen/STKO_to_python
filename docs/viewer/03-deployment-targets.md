# Deployment Targets

Where the viewer runs, what works in each setting, and how to install.
This is the document to point engineers at when they ask "can I run this
on my cluster?"

---

## 1. Target matrix

| Target | Backend | Display path | Performance | Status |
|---|---|---|---|---|
| **Local Qt desktop** | `PyVistaBackend(qt_widget=...)` | Native window on workstation | Full interactive | Phase 4 |
| **Local notebook (2D)** | `MplBackend` | Inline mpl figures | Same as today | Phase 2 |
| **Local notebook (3D)** | `PyVistaBackend(jupyter=True)` | Trame/pythreejs in cell | Good for ≤ 50k elem | Phase 3 |
| **VNC/X2Go to cluster** | `PyVistaBackend(qt_widget=...)` | X server on cluster, compressed pixels back | Acceptable if cluster has GPU or `llvmpipe` | Phase 4 (no extra work) |
| **SSH + X11 forwarding** | n/a | X11 indirect rendering | **Unusable** — VTK ≥ 9 needs OpenGL ≥ 3.2 core profile | Not supported |
| **Headless / batch** | `PyVistaBackend(off_screen=True)` | None — renders to memory framebuffer | High throughput | Phase 5 |
| **Trame web** | `PyVistaBackend(trame=True)` | Browser over WebSocket | Good with client-side `vtk.js`; OK with server-side encoded frames | Phase 6 (deferred) |

---

## 2. The local Qt desktop — the default

The "full experience" target. One process, one main window, embedded
PyVista viewport.

**Install:**

```bash
pip install "stko_to_python[viewer]"
```

This adds: `pyvista>=0.43`, `vtk>=9.2`, `PySide6>=6.5`,
`pyvistaqt>=0.11`, `qtpy`. About 200 MB on disk.

**Launch:**

```bash
mpco-viewer path/to/results.mpco
```

or from Python:

```python
from STKO_to_python import MPCODataSet
ds = MPCODataSet("path/to/results.mpco")
ds.viewer()                # blocks until window closed
ds.viewer(blocking=False)  # spawns a separate process (later phase)
```

**Platform notes:**

- **Linux:** works on X11 and Wayland (PySide6 6.5+ handles Wayland).
  GPU helps; software rendering via Mesa `llvmpipe` works but is slow
  on dense meshes.
- **macOS:** Apple Silicon + recent VTK works. Bench on M-series before
  declaring Phase 3 done.
- **Windows:** ships with OpenGL via WDDM. Tested.

---

## 3. Notebook (2D matplotlib)

This is what `STKO_to_python` already does. Nothing changes.

```bash
pip install stko_to_python
```

```python
ds.plot.deformed_shape(step=100, scale=50)
ds.plot.mesh_with_contour(component="VonMises", step=100)
```

After Phase 2, the same calls are powered by `Scene` + `MeshLayer` +
`MplBackend` under the hood; users don't notice.

---

## 4. Notebook (3D PyVista)

For interactive 3D inside Jupyter without a Qt window.

**Install:**

```bash
pip install "stko_to_python[viewer-3d]"
```

This adds PyVista + VTK but **no Qt**. Lighter than `[viewer]`.

**Usage:**

```python
ds = MPCODataSet("results.mpco")
scene = ds.scene(backend="pyvista", jupyter=True)
scene.add(MeshLayer())
scene.add(ContourLayer(component="VonMises", step=100))
scene.show()  # renders inline in the notebook cell
```

Under the hood this uses PyVista's `jupyter_backend="trame"` (the
default since pyvista 0.40) — which means even the notebook 3D path
quietly uses a tiny embedded Trame server. Works fine on `localhost`;
do not point your browser at `0.0.0.0`.

---

## 5. SSH + X11 forwarding — *don't*

We documented this in the planning conversation. Repeating it here so
the answer lives next to the install docs:

```bash
ssh -X user@cluster
mpco-viewer ...
```

What happens:

- Qt UI loads but is laggy (~30–80 ms per event over the wire).
- VTK fails to create a GL 3.2 core context over X11 indirect
  rendering, or falls back to software emulation on the client side
  and is unusable for picking and animation.

**Don't bother.** Use one of the alternatives below.

---

## 6. VNC / X2Go / NoMachine — works, with a caveat

VNC runs an X server on the cluster login node. Your laptop receives
only compressed pixels — much better than X11 forwarding.

**Requires:**

- An X server on the cluster (the admin's call).
- Either a GPU with display drivers on the cluster, **or** Mesa with
  `llvmpipe` for software OpenGL. Compute nodes usually have neither;
  this only works on login nodes or interactive nodes that admins have
  configured for graphics.
- The cluster admin to have enabled VNC/X2Go (rare on locked-down
  clusters).

**If it's available:**

```bash
# On laptop:
vncviewer cluster:1
# Inside the VNC session:
pip install "stko_to_python[viewer]"
mpco-viewer results.mpco
```

Performance: usable for inspection, less smooth than local. Animation
playback over VNC is rough; export an MP4 with the headless CLI instead.

---

## 7. Headless / batch — the cluster-native answer

When you're on the cluster and you want artifacts (images, animations,
reports), don't fight for a display. Use the headless CLI.

**Install:**

```bash
pip install "stko_to_python[viewer-headless]"
```

Adds: `pyvista`, `vtk`, `imageio`, `imageio-ffmpeg`. **No Qt.** No
display libraries beyond what PyVista needs for off-screen rendering.

**On Linux, VTK needs an OpenGL context even off-screen.** You need one of:

- An X server: `xvfb-run` gives PyVista a virtual framebuffer
  (`apt install xvfb`).
- EGL (`libEGL`, from Mesa or the GPU driver); no X needed.
- OSMesa (`libOSMesa`), for pure software rendering.

Mesa's `llvmpipe` does the rendering when there is no GPU. With none of
these, `PyVistaBackend.snapshot` / `save` raise a `RuntimeError` instead
of letting VTK segfault; if that check is wrong for your setup, set
`STKO_SKIP_GL_CHECK=1`.

**Use:**

```bash
# Single frame:
mpco-viewer screenshot results.mpco --step 250 --out frame.png

# Animation:
mpco-viewer animate results.mpco --config view.toml --out anim.mp4

# Batch render every Nth step:
mpco-viewer batch results.mpco --config batch.toml --out frames/

# Save a scene in the GUI, then run it on the cluster:
# (laptop) Edit scene visually, File > Save scene to view.toml
# (laptop) scp view.toml cluster:~/run/
# (cluster) mpco-viewer animate results.mpco --config view.toml --out anim.mp4
# (laptop) scp cluster:~/run/anim.mp4 .
```

This is the workflow most engineers actually use for big runs.

**SLURM stanza (Linux cluster, no GPU):**

```bash
#!/bin/bash
#SBATCH --partition=compute
#SBATCH --time=00:30:00
#SBATCH --mem=8G
module load python/3.12
source ~/venvs/stko/bin/activate   # has stko_to_python[viewer-headless] installed
xvfb-run -a mpco-viewer animate $RESULTS --config view.toml --out anim.mp4
```

---

## 8. Trame web — the SSH-only answer (Phase 6, deferred)

When the cluster has no VNC, no GUI, and you still want interactive 3D
without copying gigabytes home — Trame is the standard answer.

**Architecture:**

```
laptop browser ───── http (WebSocket) ─────► trame server (cluster login node)
                                                     │
                                                     └── PyVistaBackend(off_screen=True)
                                                            renders frames
                                                            → encodes h264 (server-side)
                                                            → or ships VTK polydata (client-side)
```

**One-port SSH tunnel:**

```bash
ssh -L 8080:localhost:8080 user@cluster
# In another terminal:
ssh user@cluster
mpco-viewer web results.mpco --port 8080
# In laptop browser:
open http://localhost:8080
```

**Two rendering modes:**

- **Server-side rendering:** VTK renders on the cluster, h264-encoded
  frames stream to the browser. Best when the network is fast and the
  cluster has a GPU.
- **Client-side rendering (`vtk.js`):** Server ships the
  `UnstructuredGrid` once, browser renders. Best for slow SSH links;
  picking and rotation feel native.

The CLI defaults to client-side rendering when the dataset is small
enough to fit in browser memory, server-side otherwise. Configurable.

**Status:** Phase 6, deferred. Not committed until Phase 5 is shipping.

---

## 9. Decision tree

```
                           Where am I working?
                                   │
              ┌────────────────────┼────────────────────┐
           workstation         laptop + SSH         CI / batch
              │                    │                    │
              │                    │                    │
   [viewer] + local      What I want?              [viewer-headless]
   mpco-viewer ...           │                     mpco-viewer animate ...
                             ├── interactive 3D
                             │      │
                             │      ├── cluster has VNC?
                             │      │     ├── yes → use VNC + [viewer]
                             │      │     └── no  → wait for Phase 6 (Trame)
                             │      │                or use VS Code Remote
                             │      │                   tunnels + [viewer-3d]
                             │      │
                             │      └── small data (< 100 MB)?
                             │             ├── yes → rsync home + [viewer]
                             │             └── no  → headless CLI on cluster
                             │
                             └── batch artifacts (PNG/MP4)
                                    └── [viewer-headless] + scp/rsync result
```

---

## 10. Install matrix (summary)

| Extra | Adds | Use when |
|---|---|---|
| *(none)* | nothing — base library | Notebook 2D, CI parsing tests, library consumers. |
| `[viewer-3d]` | `pyvista`, `vtk` | Notebook 3D, off-screen rendering inside Python scripts. |
| `[viewer-headless]` | `[viewer-3d]` + `imageio`, `imageio-ffmpeg` | Headless CLI on a cluster. |
| `[viewer]` | `[viewer-3d]` + `PySide6`, `pyvistaqt`, `qtpy` | Full desktop GUI on a workstation. |
| `[viewer-web]` *(Phase 6)* | `[viewer-3d]` + `trame`, `trame-vtk`, `trame-vuetify` | Browser-based viewer over SSH tunnel. |

`[viewer]` does **not** imply `[viewer-headless]` and vice versa.
A workstation install gets `[viewer]`; a cluster install gets
`[viewer-headless]`. Some users will want both — that's fine,
`pip install "stko_to_python[viewer,viewer-headless]"` is the spelling.

---

## 11. What we do **not** support

- SSH + X11 forwarding for the Qt viewer. Documented above; mentioned
  again here because someone will try it.
- Multi-process Qt viewer with a remote VTK server. Not in scope.
- Custom display protocols beyond Trame (no VNC over WebSocket, no
  custom RDP integration, etc.).
- VTK without OpenGL. PyVista and VTK 9+ assume OpenGL ≥ 3.2 core
  profile. Pure-CPU VTK builds exist but are not supported by us.

---

## 12. Quick troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError: Could not create GL context` on launch | No display, no Xvfb, no EGL. | Install `xvfb` and run under `xvfb-run`, or set `PYVISTA_OFF_SCREEN=true`. |
| Black viewport, layers don't appear | Layer attached but not made visible, or scene bounds not set. | `scene.show()` calls `set_bounds(model_bbox)` automatically; if you build a scene manually, do that yourself. |
| Animation runs at 1 fps over SSH+X | X11 indirect rendering hates this. | Don't use SSH+X. Switch to headless CLI or VNC. |
| `qt.qpa.plugin: Could not find the Qt platform plugin "xcb"` | Linux container without X libs. | `apt install libxcb-cursor0 libxkbcommon-x11-0 libdbus-1-3` or run under Xvfb. |
| Picking laggy in Qt | Could be missing the box-pick projection optimization. | Confirm `viewer/math/picking.py` is what's being called, not a naive per-point loop. |
| MP4 export fails on cluster | `ffmpeg` missing. | `imageio-ffmpeg` ships its own; if it's still failing, install system `ffmpeg`. |
