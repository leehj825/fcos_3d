import torch
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R

def calculate_intersection_volume_3d(box_a, box_b):
    #print("Calculating intersection volume between Box A and Box B.")
    #print("box_a", box_a)
    #print("box_b", box_b)
    ax_min, ay_min, az_min = box_a['location_3d']
    ax_max = ax_min + box_a['dimension_3d'][0]
    ay_max = ay_min + box_a['dimension_3d'][1]
    az_max = az_min + box_a['dimension_3d'][2]

    bx_min, by_min, bz_min = box_b['location_3d']
    bx_max = bx_min + box_b['dimension_3d'][0]
    by_max = by_min + box_b['dimension_3d'][1]
    bz_max = bz_min + box_b['dimension_3d'][2]

    #print(f"Box A Bounds: X[{ax_min}, {ax_max}], Y[{ay_min}, {ay_max}], Z[{az_min}, {az_max}]")
    #print(f"Box B Bounds: X[{bx_min}, {bx_max}], Y[{by_min}, {by_max}], Z[{bz_min}, {bz_max}]")

    intersect_min_x = max(ax_min, bx_min)
    intersect_min_y = max(ay_min, by_min)
    intersect_min_z = max(az_min, bz_min)
    intersect_max_x = min(ax_max, bx_max)
    intersect_max_y = min(ay_max, by_max)
    intersect_max_z = min(az_max, bz_max)

    #print(f"Intersection Bounds: X[{intersect_min_x}, {intersect_max_x}], Y[{intersect_min_y}, {intersect_max_y}], Z[{intersect_min_z}, {intersect_max_z}]")

    if intersect_max_x > intersect_min_x and intersect_max_y > intersect_min_y and intersect_max_z > intersect_min_z:
        intersection_volume = (intersect_max_x - intersect_min_x) * (intersect_max_y - intersect_min_y) * (intersect_max_z - intersect_min_z)
        #print(f"Intersection Volume: {intersection_volume}")
        return intersection_volume
    else:
        #print("No intersection between boxes.")
        return 0

def compute_volume_3d(box):
    # Assuming box['dimension_3d'] contains [width, height, depth]
    volume = box['dimension_3d'][0] * box['dimension_3d'][1] * box['dimension_3d'][2]
    #print(f"Volume of Box: {volume} (Width: {box['dimension_3d'][0]}, Height: {box['dimension_3d'][1]}, Depth: {box['dimension_3d'][2]})")
    return volume

def compute_iou_3d(pred_box, gt_box):
    #print("Computing IoU between a predicted box and a ground truth box.")
    intersection_volume = calculate_intersection_volume_3d(pred_box, gt_box)
    pred_volume = compute_volume_3d(pred_box)
    gt_volume = compute_volume_3d(gt_box)
    union_volume = pred_volume + gt_volume - intersection_volume
    iou = intersection_volume / union_volume if union_volume > 0 else 0
    #print(f"Computed IoU: {iou} (Pred Volume: {pred_volume}, GT Volume: {gt_volume}, Union Volume: {union_volume})")
    return iou

def compute_let_iou_3d(pred_box, gt_box, longitudinal_tolerance):
    # Assume the longitudinal_tolerance is defined along the Z-axis
    pred_box_shifted = shift_pred_box(pred_box, gt_box, longitudinal_tolerance)
    intersection_volume = calculate_intersection_volume_3d(pred_box_shifted, gt_box)
    pred_volume = compute_volume_3d(pred_box)
    gt_volume = compute_volume_3d(gt_box)
    union_volume = pred_volume + gt_volume - intersection_volume
    let_iou = intersection_volume / union_volume if union_volume > 0 else 0
    #print(f"Computed LET IoU: {let_iou} (Pred Volume: {pred_volume}, GT Volume: {gt_volume}, Union Volume: {union_volume})")
    return let_iou

def shift_pred_box(pred_box, gt_box, longitudinal_tolerance):
    # Shift the predicted box along the Z-axis to align with the ground truth box within a tolerance limit
    pred_box_shifted = pred_box.copy()
    z_shift = min(max(gt_box['location_3d'][2] - pred_box['location_3d'][2], -longitudinal_tolerance), longitudinal_tolerance)
    pred_box_shifted['location_3d'][2] += z_shift
    return pred_box_shifted


def compute_precision_recall(predictions, ground_truths, iou_threshold):
    #print(f"Computing precision and recall for IoU threshold: {iou_threshold}")
    tp, fp, fn = 0, 0, 0
    matched = set()
    for i, pred_box in enumerate(predictions):
        best_iou, best_gt_idx = 0, -1
        #print(f"Processing prediction box {i}")
        for j, gt_box in enumerate(ground_truths):
            iou = compute_iou_3d(pred_box[0], gt_box[0])
            #iou = compute_let_iou_3d(pred_box[0], gt_box[0], 5.0)
            #print(f"  IoU with GT box {j}: {iou}")
            if iou > best_iou:
                best_iou, best_gt_idx = iou, j
        if best_iou > iou_threshold and best_gt_idx not in matched:
            #print(f"  True Positive: Prediction {i} matched with GT {best_gt_idx}")
            tp += 1
            matched.add(best_gt_idx)
        else:
            #print(f"  False Positive: Prediction {i}")
            fp += 1
    fn = len(ground_truths) - len(matched)
    #print(f"  False Negatives: {fn}")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    #print(f"Precision: {precision}, Recall: {recall}")
    return precision, recall

def compute_average_precision(precisions, recalls):
    """
    Compute average precision (AP) from arrays of precision and recall.
    Assumes linear interpolation of precision.
    """
    # Ensure precisions and recalls are numpy arrays
    precisions = np.array(precisions)
    recalls = np.array(recalls)

    # Sort by recall
    sorted_indices = np.argsort(recalls)
    sorted_recall = recalls[sorted_indices]
    sorted_precision = precisions[sorted_indices]

    # Compute AP as the area under the precision-recall curve
    ap = 0.0
    for i in range(1, len(sorted_recall)):
        delta_recall = sorted_recall[i] - sorted_recall[i - 1]
        ap += delta_recall * (sorted_precision[i] + sorted_precision[i - 1]) / 2

    return ap


def compute_map(dataset_predictions, dataset_ground_truths, iou_thresholds):
    map_values_by_threshold = 0.0
    all_precisions = []
    all_recalls = []

    organized_predictions = [[pred] for pred in dataset_predictions]
    organized_ground_truths = [[gt] for gt in dataset_ground_truths]
    
    for iou_threshold in iou_thresholds:
        #print("organized_predictions", organized_predictions)
        #print("organized_ground_truths", organized_ground_truths)
        precision, recall = compute_precision_recall(organized_predictions, organized_ground_truths, iou_threshold)
        print(f"iou_threshold: {iou_threshold}, precision: {precision:.3f}, recall: {recall:.3f}")
        all_precisions.append(precision)
        all_recalls.append(recall)

    map_values_by_threshold = compute_average_precision(all_precisions, all_recalls)
    return map_values_by_threshold

# Example Usage
# Suppose you have lists of predicted boxes and ground truth boxes:
# predictions = [dict with location_3d, dimension_3d, orientation for each box]
# ground_truths = [same as above]
# You can call compute_map like this:
# map_value = compute_map(predictions, ground_truths, [0.5, 0.75])
