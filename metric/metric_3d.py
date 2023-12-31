import torch
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R

# Reference: https://github.com/google-research-datasets/Objectron/blob/master/objectron/dataset/box.py

import numpy as np
from scipy.spatial.transform import Rotation as R

# Constants for the Box class
EDGES = (
    [1, 5], [2, 6], [3, 7], [4, 8],  # lines along x-axis
    [1, 3], [5, 7], [2, 4], [6, 8],  # lines along y-axis
    [1, 2], [3, 4], [5, 6], [7, 8]   # lines along z-axis
)
FACES = np.array([
    [5, 6, 8, 7],  # +x on yz plane
    [1, 3, 4, 2],  # -x on yz plane
    [3, 7, 8, 4],  # +y on xz plane = top
    [1, 2, 6, 5],  # -y on xz plane
    [2, 4, 8, 6],  # +z on xy plane = front
    [1, 5, 7, 3],  # -z on xy plane
])
NUM_KEYPOINTS = 9

class Box:
    def __init__(self, vertices=None):
        if vertices is None:
            vertices = self.scaled_axis_aligned_vertices(np.array([1., 1., 1.]))

        self._vertices = vertices
        self._rotation = None
        self._translation = None
        self._scale = None
        self._transformation = None
        self._volume = None

    @classmethod
    def from_transformation(cls, rotation, translation, scale):
        # Check if rotation is a tensor with a single value and convert it to float
        if isinstance(rotation, torch.Tensor):
            if rotation.numel() != 1:
                raise ValueError('Rotation tensor should have a single element.')
            rotation = rotation.item()  # Convert tensor to a single float value

        # Check if rotation is a float or int
        if not isinstance(rotation, (float, int)):
            raise ValueError('Rotation should be a float or int value representing rotation around Y-axis.')

        # Construct the rotation matrix
        c, s = np.cos(rotation), np.sin(rotation)
        rotation_matrix = np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])

        scaled_identity_box = cls.scaled_axis_aligned_vertices(scale)
        vertices = np.zeros((NUM_KEYPOINTS, 3))

        # Convert translation to numpy array if it's not
        if not isinstance(translation, np.ndarray):
            translation = np.array(translation)

        scaled_identity_box = cls.scaled_axis_aligned_vertices(scale)
        vertices = np.zeros((NUM_KEYPOINTS, 3))

        translation = translation.flatten()  # Flatten the array

        for i in range(NUM_KEYPOINTS):
            rotated_point = np.matmul(rotation_matrix, scaled_identity_box[i, :])
            vertices[i, :] = rotated_point + translation

        return cls(vertices=vertices)


    @classmethod
    def scaled_axis_aligned_vertices(cls, scale):
        w = scale[0] / 2.
        h = scale[1] / 2.
        d = scale[2] / 2.
        aabb = np.array([[0., 0., 0.], [-w, -h, -d], [-w, -h, +d], [-w, +h, -d],
                         [-w, +h, +d], [+w, -h, -d], [+w, -h, +d], [+w, +h, -d],
                         [+w, +h, +d]])
        return aabb

    @classmethod
    def fit(cls, vertices):
        orientation = np.identity(3)
        translation = np.zeros((3, 1))
        scale = np.zeros(3)
        for axis in range(3):
            for edge_id in range(4):
                begin, end = EDGES[axis * 4 + edge_id]
                scale[axis] += np.linalg.norm(vertices[begin, :] - vertices[end, :])
            scale[axis] /= 4.

        x = cls.scaled_axis_aligned_vertices(scale)
        system = np.concatenate((x, np.ones((NUM_KEYPOINTS, 1))), axis=1)
        solution, _, _, _ = np.linalg.lstsq(system, vertices, rcond=None)
        orientation = solution[:3, :3].T
        translation = solution[3, :3]
        return orientation, translation, scale

    def inside(self, point):
        inv_trans = np.linalg.inv(self.transformation)
        scale = self.scale
        point_w = np.matmul(inv_trans[:3, :3], point) + inv_trans[:3, 3]
        for i in range(3):
            if abs(point_w[i]) > scale[i] / 2.:
                return False
        return True

    def sample(self):
        point = np.random.uniform(-0.5, 0.5, 3) * self.scale
        point = np.matmul(self.rotation, point) + self.translation
        return point

    @property
    def vertices(self):
        return self._vertices

    @property
    def rotation(self):
        if self._rotation is None:
            self._rotation, self._translation, self._scale = self.fit(self._vertices)
        return self._rotation

    @property
    def translation(self):
        if self._translation is None:
            self._rotation, self._translation, self._scale = self.fit(self._vertices)
        return self._translation

    @property
    def scale(self):
        if self._scale is None:
            self._rotation, self._translation, self._scale = self.fit(self._vertices)
        return self._scale

    @property
    def volume(self):
        if self._volume is None:
            i = self._vertices[2, :] - self._vertices[1, :]
            j = self._vertices[3, :] - self._vertices[1, :]
            k = self._vertices[5, :] - self._vertices[1, :]
            sys = np.array([i, j, k])
            self._volume = abs(np.linalg.det(sys))
        return self._volume

    @property
    def transformation(self):
        if self._rotation is None:
            self._rotation, self._translation, self._scale = self.fit(self._vertices)
        if self._transformation is None:
            self._transformation = np.identity(4)
            self._transformation[:3, :3] = self._rotation
            self._transformation[:3, 3] = self._translation
        return self._transformation

def compute_iou_3d(pred_box, gt_box):
    pred_box_3d = Box.from_transformation(pred_box['orientation'], pred_box['location_3d'], pred_box['dimension_3d'])
    gt_box_3d = Box.from_transformation(gt_box['orientation'], gt_box['location_3d'], gt_box['dimension_3d'])

    num_samples = 1000  # Number of samples for Monte Carlo estimation
    batch_size = 500    # Number of samples to process at a time

    intersection = 0

    for _ in range(0, num_samples, batch_size):
        # Sample points in batches and convert to PyTorch tensors
        points = torch.stack([torch.from_numpy(pred_box_3d.sample()).float() for _ in range(batch_size)])

        # Check if these points are inside the gt_box_3d
        inside_points = torch.tensor([gt_box_3d.inside(point.numpy()) for point in points])
        intersection += inside_points.sum().item()

    intersection_volume = intersection / num_samples * pred_box_3d.volume
    union_volume = pred_box_3d.volume + gt_box_3d.volume - intersection_volume
    iou = intersection_volume / union_volume if union_volume != 0 else 0

    return iou


# Function to match predictions to ground truths and compute precision and recall
def compute_precision_recall(predictions, ground_truths, iou_threshold):
    tp, fp, fn = 0, 0, 0
    #print("Computing precision and recall with IoU threshold:", iou_threshold)

    matched = set()
    for i, pred_box in enumerate(predictions):
        best_iou = 0
        best_gt_idx = -1
        for j, gt_box in enumerate(ground_truths):
            iou = compute_iou_3d(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        #print(f"Prediction {i}: Best matching GT index: {best_gt_idx}, Best IoU: {best_iou}")

        if best_iou > iou_threshold:
            #print(f"  Prediction {i} is a match (IoU > {iou_threshold})")
            if best_gt_idx not in matched:
                tp += 1
                matched.add(best_gt_idx)
                #print(f"  Added as TP, total TP: {tp}")
            else:
                fp += 1
                #print(f"  Added as FP (already matched GT), total FP: {fp}")
        else:
            fp += 1
            #print(f"  Added as FP (IoU <= {iou_threshold}), total FP: {fp}")

    fn = len(ground_truths) - len(matched)
    #print(f"Total FN (unmatched GT): {fn}")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    print("Computed precision:", precision)
    print("Computed recall:", recall)

    return precision, recall


def compute_ap(predictions, ground_truths, iou_threshold):
    # Sort predictions by confidence
    #predictions.sort(key=lambda x: x['confidence'], reverse=True)

    tp = np.zeros(len(predictions))
    fp = np.zeros(len(predictions))
    total_gt = len(ground_truths)
    used_gt = set()  # To keep track of used ground truth boxes

    for i, pred in enumerate(predictions):
        highest_iou = 0
        best_gt_idx = -1
        for j, gt in enumerate(ground_truths):
            iou = compute_iou_3d(pred, gt)  # Your IoU calculation function
            if iou > highest_iou:
                highest_iou = iou
                best_gt_idx = j

        if highest_iou >= iou_threshold and best_gt_idx not in used_gt:
            tp[i] = 1
            used_gt.add(best_gt_idx)
        else:
            fp[i] = 1

    # Calculate cumulative precision and recall
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    precision = cum_tp / (cum_tp + cum_fp)
    recall = cum_tp / total_gt

    print("Computed precision:", precision)
    print("Computed recall:", recall)

    # Interpolate precision and compute AP
    return np.trapz(np.maximum.accumulate(precision), recall)

# Function to compute mAP
def compute_map(predictions, ground_truths, iou_thresholds):
    #print("Computing mAP for IoU thresholds:", iou_thresholds)
    map_values_by_threshold = {}

    for iou_threshold in iou_thresholds:
        ap = compute_ap(predictions, ground_truths, iou_threshold)

        # Since precision and recall are now computed for the entire set,
        # they don't need to be sorted and averaged separately for each threshold
        # Instead, we directly use the precision and recall values computed
        #ap = precision * recall  # This is a simplified AP calculation for demonstration

        map_values_by_threshold[iou_threshold] = ap

    #print("Computed mAP by thresholds:", map_values_by_threshold)
    return map_values_by_threshold

# Example Usage
# Suppose you have lists of predicted boxes and ground truth boxes:
# predictions = [dict with location_3d, dimension_3d, orientation for each box]
# ground_truths = [same as above]
# You can call compute_map like this:
# map_value = compute_map(predictions, ground_truths, [0.5, 0.75])