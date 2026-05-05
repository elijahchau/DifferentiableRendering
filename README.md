Differentiable Rendering
=======================

This small project demonstrates a toy differentiable renderer that composes
many parametric shapes to approximate a target image by gradient-based
optimization. It includes a Flask frontend for uploading a reference image
and running the optimization; the app saves the intermediate frames and a
GIF so you can inspect the reconstruction process.

Repository layout
-----------------
- `differentiable_rendering.py` — core renderer and optimizer.
- `app.py` — simple Flask app that exposes an upload UI and runs the optimizer.
- `templates/index.html` — HTML UI used by the Flask app.
- `requirements.txt` — Python dependencies.
- `static/` — generated output (GIF, frames, start/final PNGs) placed here.
- `uploads/` — uploaded reference images saved here by the dev server.

Quick start
-----------
1. Install dependencies:

```bash
pip install -r "Differentiable Rendering/requirements.txt"
```

2. Run the Flask app:

```bash
python "Differentiable Rendering/app.py"
```

3. Open http://127.0.0.1:5000 in a browser, upload an image, and click "Run".

Important files and functions
-----------------------------

`differentiable_rendering.py`
- `load_image(path, size=128)`
  - Loads an image from disk, resizes to `size x size`, converts to RGB,
    normalizes to [0,1], and returns a torch tensor with shape (H, W, 3).

- `create_grid(H, W, device)`
  - Builds normalized coordinate grids `X, Y` in the range [0,1] suitable for
    evaluating analytical shapes. `X` and `Y` have shape (H, W).

- `stable_sigmoid(x)`
  - A numerically-stable sigmoid-like function used to turn distance fields
    into differentiable soft masks. It rescales and clamps inputs to avoid
    overflow and clamps outputs into (1e-6, 1 - 1e-6) to keep gradients stable.

- `render(center, dims, color, X, Y, shape_type)`
  - Renders `N` shapes into an image using the provided parameters.
  - `center` shape: (2, N) with normalized centers.
  - `dims` shape: (2, N) width/height-like parameters.
  - `color` shape: (3, N) RGB values in [0,1].
  - `X, Y` are grids from `create_grid`.
  - `shape_type` may be 'circle', 'rectangle', or 'gaussians'.
  - Returns an image tensor of shape (H, W, 3) with values clipped to [0,1].

- `optimize(target, shape_type='circle', num_shapes=50, iters=200, save_gif=True, out_prefix='optimization')`
  - Runs gradient-based optimization (Adam) to fit `num_shapes` to the `target`.
  - Saves intermediate frames and a GIF when `save_gif` is True.
  - Files written (when `out_prefix` is 'static/abcd'):
    - `static/abcd.gif`
    - `static/abcd_frames/frame_0000.png` ...
    - `static/abcd_start.png`
    - `static/abcd_final.png`
  - Returns a dict containing:
    - `'start_tensor'`: the starting rendered tensor
    - `'final_tensor'`: the final rendered tensor
    - `'paths'`: dict with saved file paths and `num_frames`

Notes and troubleshooting
-------------------------
- The optimization is compute-heavy. For reasonable interactive performance,
  use small images (e.g. 64x64 or 128x128), fewer shapes, and fewer iterations.
- If you run the Flask app in development mode it serves files written to the
  `static/` folder. Ensure `out_prefix` points into that folder if you want
  the web UI to be able to load the generated images/GIF.
- If GIFs or frames appear missing after a run, check the console output for
  `Saved animation → ...` messages and ensure the files exist in the
  `static/` directory.

Extending or improving
----------------------
- Add batching and asynchronous job handling so the web UI does not block
  while optimization runs. For production use, run the optimizer in a
  background worker (Celery, RQ) and show progress via polling or WebSockets.
- Improve rendering primitives (antialiasing, blending), adopt a differentiable
  rasterizer, or replace the simple loss with perceptual losses for better
  visual fidelity.

License & attribution
---------------------
This is an educational demo. Reuse as you like for learning and experimentation.

