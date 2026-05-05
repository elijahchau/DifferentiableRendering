"""
Differentiable rendering helper functions and optimizer.

This module provides a simple differentiable renderer that composes many
parametric shapes (circles, rectangles, or gaussian blobs) to approximate
an input target image. It is intentionally small and educational: the
renderer uses analytically differentiable masks and optimizes shape
parameters with Adam to match a target image.

Key capabilities:
- load and preprocess images
- build a normalized pixel grid
- render parametric shapes (circle, rectangle, gaussians)
- run a simple optimization loop that saves intermediate frames and a GIF

Files produced by `optimize` are named using the provided `out_prefix`.
For example, with out_prefix 'static/abcd' the function will write:
- 'static/abcd.gif' (animation)
- 'static/abcd_frames/frame_XXXX.png' (individual frames)
- 'static/abcd_start.png' (initial render)
- 'static/abcd_final.png' (final reconstruction)

Do not import or require any typing annotations in this file: docstrings
describe interfaces and expected shapes instead.
"""

import torch
import torch.nn.functional as F
import matplotlib

# Use non-interactive backend to avoid tkinter errors when saving figures
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import imageio
import os


# -----------------------------
# Load + preprocess image
# -----------------------------
def load_image(path, size=64):
    """
    Load an image from disk and return a float32 tensor in [0,1].

    If `size` is an integer the image will be resized to (size, size).
    If `size` is None the original image size is preserved.

    Returns a tensor with shape (H, W, 3).
    """
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size))
    img = np.array(img) / 255.0
    return torch.tensor(img, dtype=torch.float32)


# -----------------------------
# Grid
# -----------------------------
def create_grid(H, W, device):
    """
    Create a normalized pixel grid in [0,1]x[0,1].

    Parameters
    - H: number of rows (height)
    - W: number of columns (width)
    - device: torch device to allocate the grid on (cpu or cuda)

    Returns
    - X, Y tensors each with shape (H, W) containing the x and y
        coordinates in normalized [0,1] range.
    """
    y = torch.linspace(0, 1, H, device=device)
    x = torch.linspace(0, 1, W, device=device)
    Y, X = torch.meshgrid(y, x, indexing="ij")
    return X, Y


# -----------------------------
# Stable sigmoid
# -----------------------------
def stable_sigmoid(x):
    """
    Numerically stable sigmoid-like function used to produce crisp
    shape masks.

    The function rescales the input and clamps it to avoid overflow
    in the exponential, then applies the logistic transform and clamps
    the output away from exact 0/1 to keep gradients stable.

    Parameters
    - x: input tensor (arbitrary shape)

    Returns
    - tensor of same shape as x with values in (1e-6, 1 - 1e-6)
    """
    x = torch.clamp(20 * (1 - x), -20, 20)
    y = 1 / (1 + torch.exp(-x))
    return torch.clamp(y, 1e-6, 1 - 1e-6)


# -----------------------------
# Renderer
# -----------------------------
def render(center, dims, color, X, Y, shape_type):
    """
    Render N shapes into an image given shape parameters.

    Parameters
    - center: tensor of shape (2, N) giving normalized (x,y) centers in [0,1]
    - dims: tensor of shape (2, N) giving width/height parameters (unitless)
    - color: tensor of shape (3, N) giving RGB colors in [0,1]
    - X, Y: coordinate grids from `create_grid` with shape (H, W)
    - shape_type: one of 'circle', 'rectangle', or 'gaussians'

    Returns
    - image tensor with shape (H, W, 3) and values in [0,1]
    """
    N = center.shape[1]

    cx = center[0].view(1, 1, N)
    cy = center[1].view(1, 1, N)

    dx = dims[0].view(1, 1, N)
    dy = dims[1].view(1, 1, N)

    X = X.unsqueeze(-1)
    Y = Y.unsqueeze(-1)

    if shape_type == "circle":
        dist = ((X - cx) ** 2 / (0.5 * dx) ** 2) + ((Y - cy) ** 2 / (0.5 * dy) ** 2)
        mask = stable_sigmoid(dist)

    elif shape_type == "rectangle":
        dist = torch.maximum(
            torch.abs(X - cx) / (0.5 * dx), torch.abs(Y - cy) / (0.5 * dy)
        )
        mask = stable_sigmoid(dist)

    elif shape_type == "gaussians":
        dist = ((X - cx) ** 2 / (0.02 * dx)) + ((Y - cy) ** 2 / (0.02 * dy))
        mask = torch.exp(-dist)

    color = color.view(1, 1, 3, N)
    image = torch.sum(mask.unsqueeze(2) * color, dim=-1)

    return torch.clamp(image, 0, 1)


# -----------------------------
# Optimization with animation
# -----------------------------
def optimize(
    target,
    shape_type="circle",
    num_shapes=50,
    iters=200,
    save_gif=True,
    out_prefix="optimization",
):
    """
    Optimize a set of parametric shapes to match the `target` image.

    Parameters
    - target: a torch tensor of shape (H, W, 3) with values in [0,1]
    - shape_type: 'circle' | 'rectangle' | 'gaussians'
    - num_shapes: how many shapes to use in the reconstruction
    - iters: number of optimization iterations
    - save_gif: whether to save an animation and frames
    - out_prefix: filesystem prefix used to save outputs

    Returns
    - a dict with keys 'start_tensor', 'final_tensor', and 'paths'
        where 'paths' contains filesystem paths for 'gif', 'frames_dir',
        'start', and 'final' when available.

    Notes
    - `out_prefix` should be chosen so that the process has write
        permission. When used from the Flask app, `out_prefix` is typically
        inside the application's `static/` directory so files are served
        by the development server.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = target.to(device)
    print(device)

    H, W, _ = target.shape
    X, Y = create_grid(H, W, device)

    # Parameters
    center = torch.rand(2, num_shapes, device=device, requires_grad=True)
    dims = (0.1 + 0.4 * torch.rand(2, num_shapes, device=device)).requires_grad_()
    color = torch.rand(3, num_shapes, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([center, dims, color], lr=0.05)

    start_rendered = None
    frames_iters = []
    losses = []

    frames_dir = f"{out_prefix}_frames"
    if save_gif:
        os.makedirs(frames_dir, exist_ok=True)

    for i in range(iters):
        optimizer.zero_grad()

        rendered = render(center, dims, color, X, Y, shape_type)
        # Capture initial rendered before any optimization step
        if i == 0:
            start_rendered = rendered.detach().cpu().numpy()
        loss = F.mse_loss(rendered, target)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            center.clamp_(0.01, 0.99)
            dims.clamp_(0.01, 1.0)
            color.clamp_(0, 1)

        # Save one frame per iteration (image + synchronized loss plot)
        if save_gif:
            img = rendered.detach().cpu().numpy()
            img = (img * 255).astype(np.uint8)
            frame_path = os.path.join(frames_dir, f"frame_{i:06d}.png")
            imageio.imwrite(frame_path, img)
            frames_iters.append(i)
            # save loss plot up to this iteration
            try:
                loss_plot_path = os.path.join(frames_dir, f"loss_{i:06d}.png")
                plt.figure(figsize=(4, 2.5))
                plt.plot(losses + [loss.item()])
                plt.xlabel("Iteration")
                plt.ylabel("MSE Loss")
                plt.grid(True, alpha=0.25)
                plt.tight_layout()
                plt.savefig(loss_plot_path)
                plt.close()
            except Exception:
                pass

        if i % 20 == 0:
            print(f"Iter {i}, Loss {loss.item():.6f}")
        losses.append(loss.item())

    # Save frames directory info and numeric losses
    saved_paths = {}
    if save_gif:
        saved_paths["frames_dir"] = frames_dir
        # number of saved frames equals number of iterations
        saved_paths["num_frames"] = iters
        # save numeric loss values for UI
        try:
            import json

            loss_values_path = f"{out_prefix}_loss_values.json"
            with open(loss_values_path, "w") as f:
                json.dump(losses, f)
            saved_paths["loss_values"] = loss_values_path
        except Exception:
            pass

    # Save start and final images
    start_img = (
        start_rendered
        if start_rendered is not None
        else rendered.detach().cpu().numpy()
    )
    final_img = rendered.detach().cpu().numpy()

    start_path = f"{out_prefix}_start.png"
    final_path = f"{out_prefix}_final.png"
    imageio.imwrite(start_path, (start_img * 255).astype(np.uint8))
    imageio.imwrite(final_path, (final_img * 255).astype(np.uint8))
    saved_paths["start"] = start_path
    saved_paths["final"] = final_path

    # save loss curve
    try:
        loss_path = f"{out_prefix}_loss.png"
        plt.figure(figsize=(6, 3))
        plt.plot(losses)
        plt.xlabel("Iteration")
        plt.ylabel("MSE Loss")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(loss_path)
        plt.close()
        saved_paths["loss"] = loss_path
    except Exception:
        pass

    return {
        "start_tensor": torch.tensor(start_img, dtype=torch.float32),
        "final_tensor": torch.tensor(final_img, dtype=torch.float32),
        "paths": saved_paths,
    }


# -----------------------------
# Run everything
# -----------------------------
if __name__ == "__main__":

    # 🔁 Change this to your image
    target = load_image("monkey.jpg", size=64)

    res = optimize(
        target,
        shape_type="gaussians",  # "circle", "rectangle", "gaussians"
        num_shapes=150,
        iters=300,
        out_prefix="optimization",
    )
