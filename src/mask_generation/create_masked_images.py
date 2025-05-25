import os
import cv2
import glob
import numpy as np
import random
import argparse
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt

from shape_synthetizer_scaled_READY import (
    generate_deformed_circle, generate_wavy_ellipses, generate_deformed_ellipse,
    generate_deformed_rectangle, generate_deformed_trapezoid,
    generate_deformed_rounded_rectangle, generate_deformed_dotted_rectangle,
    generate_deformed_arc, generate_parallel_wavy_arcs, generate_parallel_wavy_lines,
    generate_deformed_grid,
    generate_deformed_filled_patch, generate_deformed_filled_patch_multi, generate_deformed_linear_border,
    generate_parallel_linear_borders
)

MASK_TYPES = {
    "circular": {
        "functions": {
            generate_deformed_circle: 0.35,
            generate_deformed_ellipse: 0.65,
        }
    },
    "borders": {
        "functions": {
            generate_deformed_arc: 0.25,
            generate_parallel_wavy_arcs: 0.25,
            generate_deformed_linear_border: 0.25,
            generate_parallel_linear_borders: 0.25,
        }
    },
    "grids": {
        "functions": {
            generate_deformed_grid: 0.6,
            generate_deformed_dotted_rectangle: 0.4,
        }
    },
    "filled": {
        "functions": {
            generate_deformed_filled_patch: 0.2,
            generate_deformed_filled_patch_multi: 0.5,
            generate_wavy_ellipses: 0.3
        }
    },
    "rectangular": {
        "functions": {
            generate_deformed_rectangle: 0.6,
            generate_deformed_trapezoid: 0.20,
            generate_deformed_rounded_rectangle: 0.20,
        }
    }
}

def weighted_random_choice(options_dict):
    funcs, weights = zip(*options_dict.items())
    return random.choices(funcs, weights=weights, k=1)[0]

def apply_mask_to_image(image_float, mask, is_positive, alpha):
    mask_rgb = np.stack([mask] * 3, axis=-1)
    if is_positive:
        return np.clip(image_float - alpha * mask_rgb, 0.0, 1.0)
    else:
        return np.clip(image_float + alpha * mask_rgb, 0.0, 1.0)

def shift_mask(mask, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]), borderValue=0)

def main():
    parser = argparse.ArgumentParser(description="Mask generator for geoportal images")
    parser.add_argument("--positive_mask_prob", type=float, default=0.8, help="Probability of applying a positive cropmark mask (darker mask)")
    parser.add_argument("--multi_type_probs", nargs=3, type=float, default=[0.85, 0.10, 0.05], help="Probabilities for 1, 2, or 3 mask types on the same image.")
    parser.add_argument("--repeat_single_mask_prob", type=float, default=0.05, help="Probability of repeating a single mask type multiple times on the same image.")
    parser.add_argument("--mask_type_weights", nargs=5, type=float, default=[0.20, 0.20, 0.17, 0.18, 0.25], help="Relative weights for each mask type. Should sum to 1.0. Order: circular, borders, grids, filled, rectangular.")
    parser.add_argument("--alpha_random_min", type=float, default=0.10, help="Minimum alpha value for mask application. The smaller the value, the less visible and distinct the mask will be. Reccommended range is 0.05 to 0.15.")
    parser.add_argument("--alpha_random_max", type=float, default=0.15, help="Maximum alpha value for mask application. The bigger the value, the more visible and distinct the mask will be. Recommended range is 0.15 to 0.25.")
    parser.add_argument("--input_folder", type=str, default="data/geoportal_mock_dataset/negative",
                        help="Path to input folder with original images.")
    parser.add_argument("--output_folder", type=str, default="data/geoportal_mock_dataset/positive_procedural",
                        help="Path to output folder for masked images.")

    parser.add_argument("--save_masks", action="store_true", help="Save generated masks to a separate folder.")
    parser.add_argument("--plot_mask_stats", action="store_true", help="Plot distribution histogram of applied mask types.")
    parser.add_argument("--plot_examples", action="store_true", help="Plot examples of generated masks.")
    parser.add_argument("--seed", type=int, default=43, help="Random seed for reproducibility.")
    parser.add_argument("--plot_mask_examples", action="store_true", help="Plot tiled figure with examples of individual mask types. Only if save_masks is enabled.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # input_dir = "../../data/geoportal_mock_dataset/negative"
    # output_dir = "data/geoportal_mock_dataset/positive_procedural"
    # masks_dir = "data/geoportal_mock_dataset/masks"
    input_dir = args.input_folder
    if not os.path.exists(input_dir):
        raise ValueError(f"Input folder does not exist: {input_dir}")
    output_dir = args.output_folder
    masks_dir = os.path.join(os.path.dirname(output_dir), "masks")

    os.makedirs(output_dir, exist_ok=True)
    if args.save_masks:
        os.makedirs(masks_dir, exist_ok=True)

    application_stats = defaultdict(int)
    image_paths = glob.glob(os.path.join(input_dir, "*.png"))
    print(f"Processing {len(image_paths)} images...")

    counter = 0

    for image_path in image_paths:
        filename = os.path.basename(image_path)
        try:
            m_px = float(filename.split("_")[0])
        except ValueError:
            print(f"Skipping invalid filename: {filename}")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"Error reading image: {image_path}")
            continue

        image = cv2.resize(image, (350, 350))
        image_float = image.astype(np.float32) / 255.0

        # Set mask weights
        for i, t in enumerate(MASK_TYPES.keys()):
            MASK_TYPES[t]["weight"] = args.mask_type_weights[i]

        num_types = random.choices([1, 2, 3], weights=args.multi_type_probs)[0]
        selected_types = random.choices(
            list(MASK_TYPES.keys()),
            weights=[MASK_TYPES[t]["weight"] for t in MASK_TYPES],
            k=num_types
        )

        final_image = image_float.copy()
        applied_types = []

        for t_idx, mask_type in enumerate(selected_types):
            fn_dict = MASK_TYPES[mask_type]["functions"]
            selected_fn = weighted_random_choice(fn_dict)

            repeat_this_mask = (
                t_idx == 0 and num_types == 1 and random.random() < args.repeat_single_mask_prob
            )
            repeat_count = random.randint(2, 3) if repeat_this_mask else 1

            for r in range(repeat_count):
                dx = random.randint(-150, 150) if repeat_count > 1 else 0
                dy = random.randint(-150, 150) if repeat_count > 1 else 0

                mask = selected_fn(m_px)
                if mask is None:
                    mask = weighted_random_choice(fn_dict)(m_px)
                    if mask is None:
                        print(f"Skipping: {filename} (m_px={m_px})")
                        continue

                if repeat_count > 1:
                    mask = shift_mask(mask, dx, dy)

                is_positive = random.random() < args.positive_mask_prob
                alpha = random.uniform(args.alpha_random_min, args.alpha_random_max)
                final_image = apply_mask_to_image(final_image, mask, is_positive, alpha)
                application_stats[mask_type] += 1
                applied_types.append(mask_type)

                if args.save_masks:
                    mask_path = os.path.join(masks_dir, f"{filename[:-4]}_{mask_type}.png")
                    cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))

                if args.plot_examples and counter <3:
                    fig, axs = plt.subplots(1, 3, figsize=(12, 4.1))
                    axs[0].imshow(image_float, vmin=0, vmax=1)
                    axs[0].set_title("Original")
                    axs[1].imshow(mask, cmap='gray')
                    axs[1].set_title("Mask")
                    axs[2].imshow(final_image, vmin=0, vmax=1)
                    axs[2].set_title("Masked")
                    for ax in axs:
                        ax.axis('off')
                    plt.tight_layout()
                    plt.show()

                counter += 1

        type_suffix = "-".join(sorted(set(applied_types)))
        out_filename = f"{m_px}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{type_suffix}.png"
        out_path = os.path.join(output_dir, out_filename)
        cv2.imwrite(out_path, (final_image * 255).astype(np.uint8))

    if args.plot_mask_stats:
        plt.figure(figsize=(8, 5))
        types = list(application_stats.keys())
        counts = [application_stats[t] for t in types]
        plt.bar(types, counts, color='teal')
        plt.xlabel("Morphological cropmark type")
        plt.ylabel("Number of applied masks")
        plt.title("Distribution of cropmark types")
        plt.xticks(rotation=20)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    if args.save_masks and args.plot_mask_examples:

        folder_path = masks_dir
        types = ["circular", "rectangular", "borders", "grids", "filled"]
        counts = [6, 6, 6, 6, 6]
        random_seeds = [42, 666, 6, 2, 3]
        n_rows = 6
        n_cols = 5
        target_size = (256, 256)
        fallback_img = np.ones((350, 350, 3), dtype=np.uint8) * 255
        type_to_images = defaultdict(list)
        for typ in types:
            for fname in sorted(os.listdir(folder_path)):
                if fname.endswith(".png") and f"_{typ}.png" in fname:
                    type_to_images[typ].append(os.path.join(folder_path, fname))

        selected_images = []
        for typ, count, seed in zip(types, counts, random_seeds):
            imgs = type_to_images.get(typ, [])
            random.Random(seed).shuffle(imgs)
            padded_imgs = imgs[:count] + [None] * max(0, count - len(imgs))
            selected_images.append(padded_imgs)

        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 12))
        plt.subplots_adjust(wspace=0.01, hspace=0.01, left=0, right=1, top=0.94, bottom=0)
        fig.suptitle("Visual diversity of individual synthetic procedural mask types", fontsize=16, y=0.99)

        for col in range(n_cols):
            axes[0, col].set_title(types[col].capitalize(), fontsize=13)

        for row in range(n_rows):
            for col in range(n_cols):
                ax = axes[row, col]
                ax.axis("off")
                img_path = selected_images[col][row]

                if img_path is None or not os.path.exists(img_path):
                    img = fallback_img.copy()
                else:
                    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                    if img is None:
                        img = fallback_img.copy()
                    else:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                        _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
                        coords = cv2.findNonZero(thresh)
                        if coords is not None:
                            x, y, w, h = cv2.boundingRect(coords)
                            img = img[y:y + h, x:x + w]
                        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

                ax.imshow(img)

        plt.show()


if __name__ == "__main__":
    main()
