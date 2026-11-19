"""Subject-level PROMISE12 inference and metrics for the BCP U-Net."""

import csv
import json
from pathlib import Path

from medpy import metric
import numpy as np
from scipy.ndimage import zoom
from scipy import ndimage
import torch


def largest_component_3d(prediction):
    """Keep the largest 26-connected foreground component in one volume."""
    prediction = np.asarray(prediction, dtype=np.uint8)
    foreground = prediction > 0
    if not foreground.any():
        return np.zeros_like(prediction, dtype=np.uint8)
    structure = ndimage.generate_binary_structure(3, 3)
    labels, count = ndimage.label(foreground, structure=structure)
    if count == 0:
        return np.zeros_like(prediction, dtype=np.uint8)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return (labels == sizes.argmax()).astype(np.uint8)


@torch.inference_mode()
def predict_volume(model, image, patch_size=(256, 256), device="cuda"):
    """Run the official BCP-style 2D inference and restore each slice size."""
    model.eval()
    volume = image.cpu().numpy()
    prediction = np.zeros(volume.shape, dtype=np.uint8)
    for index, source_slice in enumerate(volume):
        height, width = source_slice.shape
        resized = zoom(
            source_slice,
            (patch_size[0] / height, patch_size[1] / width),
            order=0,
        )
        tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float().to(device)
        logits = model(tensor)
        output = logits.softmax(dim=1).argmax(dim=1).squeeze(0).cpu().numpy()
        prediction[index] = zoom(
            output,
            (height / patch_size[0], width / patch_size[1]),
            order=0,
        ).astype(np.uint8)
    return prediction


def binary_metrics(prediction, target, distances=True):
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    pred_count, target_count = prediction.sum(), target.sum()
    if pred_count == 0 and target_count == 0:
        return {"dice": 1.0, "jaccard": 1.0, "hd95": 0.0, "asd": 0.0}
    if pred_count == 0 or target_count == 0:
        penalty = float(np.sqrt(sum(size * size for size in target.shape)))
        return {"dice": 0.0, "jaccard": 0.0, "hd95": penalty, "asd": penalty}
    intersection = np.logical_and(prediction, target).sum()
    union = np.logical_or(prediction, target).sum()
    values = {
        "dice": float(2.0 * intersection / (pred_count + target_count)),
        "jaccard": float(intersection / union),
        "hd95": float("nan"),
        "asd": float("nan"),
    }
    if distances:
        # Unit spacing intentionally reports voxel distances.
        values["hd95"] = float(metric.binary.hd95(prediction, target))
        values["asd"] = float(metric.binary.asd(prediction, target))
    return values


def evaluate_dataset(
    model,
    dataset,
    patch_size=(256, 256),
    device="cuda",
    distances=True,
    postprocess=None,
):
    rows = []
    for index in range(len(dataset)):
        sample = dataset[index]
        prediction = predict_volume(model, sample["image"], patch_size, device)
        if postprocess == "largest_3d":
            prediction = largest_component_3d(prediction)
        values = binary_metrics(prediction, sample["label"].numpy(), distances)
        rows.append({"case": sample["case"], "slices": len(prediction), **values})
    summary = {}
    for name in ("dice", "jaccard", "hd95", "asd"):
        array = np.asarray([row[name] for row in rows], dtype=np.float64)
        valid = array[~np.isnan(array)]
        summary[f"{name}_mean"] = float(valid.mean()) if len(valid) else float("nan")
        summary[f"{name}_sd"] = float(valid.std(ddof=1)) if len(valid) > 1 else 0.0
    summary["num_cases"] = len(rows)
    summary["aggregation"] = "per-volume metrics followed by mean and sample SD"
    summary["distance_unit"] = "voxel"
    return rows, summary


def save_results(rows, summary, output_dir, prefix="test"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{prefix}_per_case.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = output_dir / f"{prefix}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    text_path = output_dir / f"{prefix}_summary.txt"
    text_path.write_text(
        "\n".join(
            [
                f"cases: {summary['num_cases']}",
                f"Dice: {summary['dice_mean']:.6f} ± {summary['dice_sd']:.6f}",
                f"Jaccard: {summary['jaccard_mean']:.6f} ± {summary['jaccard_sd']:.6f}",
                f"HD95 (voxel): {summary['hd95_mean']:.6f} ± {summary['hd95_sd']:.6f}",
                f"ASD (voxel): {summary['asd_mean']:.6f} ± {summary['asd_sd']:.6f}",
                summary["aggregation"],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return csv_path, json_path, text_path
