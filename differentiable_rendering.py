import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import imageio
import os


# -----------------------------
# Load + preprocess image
# -----------------------------
def load_image(path, size=128):
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size))
    img = np.array(img) / 255.0
    return torch.tensor(img, dtype=torch.float32)


# -----------------------------
# Grid
# -----------------------------
def create_grid(H, W, device):
    y = torch.linspace(0, 1, H, device=device)
    x = torch.linspace(0, 1, W, device=device)
    Y, X = torch.meshgrid(y, x, indexing="ij")
    return X, Y


# -----------------------------
# Stable sigmoid
# -----------------------------
def stable_sigmoid(x):
    x = torch.clamp(150 * (1 - x), -20, 20)
    y = 1 / (1 + torch.exp(-x))
    return torch.clamp(y, 1e-6, 1 - 1e-6)


# -----------------------------
# Renderer
# -----------------------------
def render(center, dims, color, X, Y, shape_type):
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = target.to(device)

    H, W, _ = target.shape
    X, Y = create_grid(H, W, device)

    # Parameters
    center = torch.rand(2, num_shapes, device=device, requires_grad=True)
    dims = torch.rand(2, num_shapes, device=device, requires_grad=True)
    color = torch.rand(3, num_shapes, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([center, dims, color], lr=0.05)

    frames = []
    start_rendered = None

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

        # Save frame every few steps
        if i % 5 == 0 or i == iters - 1:
            img = rendered.detach().cpu().numpy()
            img = (img * 255).astype(np.uint8)
            frames.append(img)

        if i % 20 == 0:
            print(f"Iter {i}, Loss {loss.item():.6f}")

    # Save GIF and individual frames
    saved_paths = {}
    if save_gif:
        gif_path = f"{out_prefix}.gif"
        frames_dir = f"{out_prefix}_frames"
        os.makedirs(frames_dir, exist_ok=True)
        # save individual frames
        for idx, img in enumerate(frames):
            frame_path = os.path.join(frames_dir, f"frame_{idx:04d}.png")
            imageio.imwrite(frame_path, img)
        # save gif
        imageio.mimsave(gif_path, frames, fps=10)
        print(f"Saved animation → {gif_path}")
        saved_paths["gif"] = gif_path
        saved_paths["frames_dir"] = frames_dir
        saved_paths["num_frames"] = len(frames)

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
    target = load_image("monkey.jpg", size=128)

    res = optimize(
        target,
        shape_type="gaussians",  # "circle", "rectangle", "gaussians"
        num_shapes=150,
        iters=300,
        out_prefix="optimization",
    )

    # Show target and final result
    plt.subplot(1, 2, 1)
    plt.imshow(target.numpy())
    plt.title("Target")

    plt.subplot(1, 2, 2)
    plt.imshow(res["final_tensor"].numpy())
    plt.title("Reconstruction")

    plt.show()
