import argparse
import os

import cv2
import numpy as np
from skimage.measure import label, regionprops

# IoU levels to determine if an instance prediction is a "True Positive"
IOU_THRESHOLDS = [0.5, 0.75]
# Score threshold for foreground extraction.
SCORE_THRESHOLD = 0.1
SEMANTIC_IOU = "semantic_iou"
DICE_COEFFICIENT = "dice_coefficient"
COUNT_GT = "count_gt"
COUNT_PRED = "count_pred"
INSTANCE_PRECISION = "instance_precision"
INSTANCE_RECALL = "instance_recall"
INSTANCE_F1 = "instance_f1"
MIOU = "mIoU"
MDICE = "mDice"


def print_eval_report(results: dict, title: str = "Evaluation Results", log_path: str = None):
    """
    Prints a formatted, easy-to-read summary of the evaluation dictionary.
    """
    width = max(len(title) + 2, 25)
    report = []
    report.append(f"\n{'=' * width}")
    report.append(f"{title.upper():^{width}}")
    report.append(f"{'-' * width}")

    for metric, value in results.items():
        # Clean up key names (e.g., 'total_mIoU' -> 'total mIou')
        display_name = metric.replace("_", " ")

        # Format floats to 4 decimal places, leave others as is
        if isinstance(value, float):
            report.append(f"{display_name:<{width - 8}}: {value:>6.4f}")
        else:
            report.append(f"{display_name:<{width - 8}}: {value:>6}")

    report.append(f"{'=' * width}\n")

    # Print to console
    full_report = "\n".join(report)
    print(full_report)

    # Save to file if path is provided
    if log_path:
        with open(log_path, "a") as f:
            f.write(full_report)


def evaluate_segmentation_performance(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    threshold: float = 255 * SCORE_THRESHOLD,
) -> dict[str, float]:
    """
    Evaluates segmentation using semantic and instance-level metrics.

    The function computes pixel-wise overlap (IoU, Dice) and instance-level
    matching (Precision, Recall) by treating connected components as
    individual segments.

    Args:
        pred_mask: A grayscale or binary NumPy array from the model output.
        gt_mask: A grayscale or binary NumPy array representing the ground truth.
        threshold: Binarization threshold for foreground extraction.

    Returns:
        A dictionary containing:
            - SEMANTIC_IOU: Global pixel-wise Intersection over Union.
            - DICE_COEFFICIENT: Global Dice score.
            - 'instance_precision_50': Precision at 0.5 IoU threshold.
            - 'instance_recall_50': Recall at 0.5 IoU threshold.
            - 'instance_f1_50': F1 Score at 0.5 IoU threshold.

    Raises:
        ValueError: If input masks have different dimensions.
    """
    # Validate input
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(f"Shape mismatch: Pred {pred_mask.shape} vs GT {gt_mask.shape}")

    # 1. Pre-processing: Binarize and Label
    pred_bin = (pred_mask > threshold).astype(np.uint8)
    gt_bin = (gt_mask > threshold).astype(np.uint8)

    # 2. Semantic Evaluation (Pixel-wise)
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    # 2.1. Iou calculation
    s_iou = intersection / union if union > 0 else 0.0
    # 2.2. Dice calculation
    dice = (
        (2 * intersection) / (pred_bin.sum() + gt_bin.sum())
        if (pred_bin.sum() + gt_bin.sum()) > 0
        else 0.0
    )

    # 3. Instance Evaluation (Object-wise)
    # label() identifies separate 'islands' of target pixels
    pred_label = label(pred_bin)
    gt_label = label(gt_bin)
    # 3.1. Calculate predicted and ground truth properties
    pred_props = regionprops(pred_label)
    gt_props = regionprops(gt_label)

    # 4. Init eval_result dictionary
    eval_result = {
        SEMANTIC_IOU: s_iou,
        DICE_COEFFICIENT: dice,
        COUNT_GT: len(gt_props),
        COUNT_PRED: len(pred_props),
    }

    # 5. Instance matching logic for each threshold
    for thresh in IOU_THRESHOLDS:
        tp = 0
        matched_gt_indices = set()

        for pred_prop in pred_props:
            best_iou = 0
            best_gt_idx = -1

            # 5.1. Create a localized mask for the current predicted property
            p_mask = pred_label == pred_prop.label

            for idx, gt_prop in enumerate(gt_props):
                # Continue for matched indexes
                if idx in matched_gt_indices:
                    continue

                # 5.2. Create a localized mask for the current ground truth property
                g_mask = gt_label == gt_prop.label

                # 5.3. Check intersection only in the bounding box area for speed
                intersection = np.logical_and(p_mask, g_mask).sum()
                union = np.logical_or(p_mask, g_mask).sum()
                iou = intersection / union if union > 0 else 0

                # 5.4. Save best iou
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            # 5.4. Count tp if best_iou >= thresh
            if best_iou >= thresh:
                tp += 1
                matched_gt_indices.add(best_gt_idx)

        # 5.5. Calculate Precision and Recall and F1 Score for this threshold
        precision = tp / len(pred_props) if len(pred_props) > 0 else 0.0
        recall = tp / len(gt_props) if len(gt_props) > 0 else 0.0
        # Calculate F1 Score
        if (precision + recall) > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0

        # 5.6. Save evaluation result to final dict
        suffix = int(thresh * 100)
        eval_result[f"{INSTANCE_PRECISION}_{suffix}"] = precision
        eval_result[f"{INSTANCE_RECALL}_{suffix}"] = recall
        eval_result[f"{INSTANCE_F1}_{suffix}"] = f1_score

    return eval_result


def evaluate_dataset(all_image_results: list[dict[str, float]]) -> dict:
    """
    Aggregates per-image evaluation results into dataset-level metrics.

    Args:
        all_image_results: A list of dictionaries, where each dictionary
            contains the output from `evaluate_segmentation_performance`.

    Returns:
        A dictionary containing dataset-wide averages and global instance scores.
    """
    # Validate input
    if not all_image_results:
        return {}

    # 1. Aggregate Semantic Metrics (Simple Mean)
    mean_iou = np.mean([r[SEMANTIC_IOU] for r in all_image_results])
    mean_dice = np.mean([r[DICE_COEFFICIENT] for r in all_image_results])

    # 2. Aggregate Instance Metrics (Global Summation)
    # We sum TP, FP, and GT counts across all images for a more stable metric
    total_gt = sum(r[COUNT_GT] for r in all_image_results)
    total_pred = sum(r[COUNT_PRED] for r in all_image_results)

    # 3. Init final_result dictionary
    final_result = {
        MIOU: mean_iou,
        MDICE: mean_dice,
        "Instance_count": total_gt,
    }
    for thresh in IOU_THRESHOLDS:
        suffix = int(thresh * 100)
        # To calculate global Precision/Recall, we need the TP count back.
        # Note: You may need to update the previous function to return 'tp_50' count.
        # For now, we'll derive it: TP = Precision * Total_Pred
        total_tp = sum(
            r[f"{INSTANCE_PRECISION}_{suffix}"] * r[COUNT_PRED] for r in all_image_results
        )

        total_precision = total_tp / total_pred if total_pred > 0 else 0.0
        total_recall = total_tp / total_gt if total_gt > 0 else 0.0

        # F1 Score is often the most useful 'single number' for wall detection
        total_f1 = (
            (2 * total_precision * total_recall) / (total_precision + total_recall)
            if (total_precision + total_recall) > 0
            else 0.0
        )

        final_result[f"Precision_{suffix}"] = total_precision
        final_result[f"Recall_{suffix}"] = total_recall
        final_result[f"F1_Score_{suffix}"] = total_f1

    return final_result


if __name__ == "__main__":

    # 1. Read inputs
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred_path", type=str, required=True, help="Path to the prediction results"
    )
    parser.add_argument(
        "--gt_path", type=str, required=True, help="Path to the ground truth masks"
    )
    args = parser.parse_args()

    # Init for next steps
    pred_root = args.pred_path
    mask_root = args.gt_path
    mask_name_list = sorted(os.listdir(mask_root))
    results = []
    # 2. Evaluate each gt file
    len_mask_name_list = len(mask_name_list)
    for i, mask_name in enumerate(mask_name_list):
        print(f"[{i+1}/{len_mask_name_list}] {mask_name}")
        # 2.1. Read gt and pred path
        gt_path = os.path.join(mask_root, mask_name)
        pred_path = os.path.join(pred_root, mask_name[:-4] + ".png")
        # 2.2. Read gt and pred images
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        # 2.3. Evaluate
        result = evaluate_segmentation_performance(pred_mask, gt_mask)
        # 2.4. Save result
        # print_eval_report(result, title="Single Evaluation")
        results.append(result)

    # 3. Evaluate all results
    final_result = evaluate_dataset(results)
    print_eval_report(final_result, title="Segmentation Evaluation")
