import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt


def load_frame_sequence(frame_dir, ext="png"):
    paths = sorted(glob.glob(os.path.join(frame_dir, f"*.{ext}")))
    if not paths:
        raise FileNotFoundError(f"No .{ext} files found in {frame_dir}")

    frames = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read {p}")
        frames.append(img)

    filenames = [os.path.basename(p) for p in paths]
    return frames, filenames


def dataset_summary(frames, filenames):
    shapes = set(f.shape for f in frames)
    dtypes = set(f.dtype for f in frames)

    print(f"Total frames        : {len(frames)}")
    print(f"First / last file   : {filenames[0]}  ->  {filenames[-1]}")
    print(f"Resolution(s) found : {shapes}")
    print(f"Dtype(s) found      : {dtypes}")

    if len(shapes) > 1:
        print("WARNING: inconsistent resolution across frames")
    else:
        h, w = next(iter(shapes))
        print(f"Resolution confirmed: {w} x {h}")

    pixel_min = min(int(f.min()) for f in frames)
    pixel_max = max(int(f.max()) for f in frames)
    print(f"Pixel value range   : [{pixel_min}, {pixel_max}]")

    return {
        "n_frames": len(frames),
        "resolution": next(iter(shapes)) if len(shapes) == 1 else None,
        "consistent_resolution": len(shapes) == 1,
        "pixel_range": (pixel_min, pixel_max),
    }


def show_pixel_matrix(frame, filename, block=5):
    print(f"\n{filename} shape: {frame.shape}")
    print(frame[:block, :block])


def plot_sample_frames(frames, filenames, n_samples=4, save_path=None):
    idxs = np.linspace(0, len(frames) - 1, n_samples, dtype=int)
    fig, axes = plt.subplots(1, n_samples, figsize=(4 * n_samples, 4))
    if n_samples == 1:
        axes = [axes]

    for ax, i in zip(axes, idxs):
        ax.imshow(frames[i], cmap="gray", vmin=0, vmax=255)
        ax.set_title(filenames[i])
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    frame_dir = "data/car-turn"
    frames, filenames = load_frame_sequence(frame_dir)
    dataset_summary(frames, filenames)
    show_pixel_matrix(frames[0], filenames[0])
    plot_sample_frames(frames, filenames, save_path="outputs/sample_frames.png")
