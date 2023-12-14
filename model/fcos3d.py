import math
import warnings
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn, Tensor

# Import from torchvision
from torchvision.ops import boxes as box_ops, generalized_box_iou_loss, misc as misc_nn_ops, sigmoid_focal_loss
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers

# For resnet and its weights, import directly from torchvision.models
#from torchvision.models import resnet50
#from torchvision.models.resnet import ResNet50_Weights
from torchvision.models import resnet101
from torchvision.models.resnet import ResNet101_Weights

# Util functions from torchvision
from torchvision.models.detection import _utils as det_utils

import metric.metric as metric

__all__ = [
    "FCOS",
    "fcos3d",
]

# Reference: https://github.com/pytorch/vision/blob/main/torchvision/models/detection/fcos.py

class FCOSHead(nn.Module):
    """
    A regression and classification head for use in FCOS.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_classes (int): number of classes to be predicted
        num_convs (Optional[int]): number of conv layer of head. Default: 4.
    """

    __annotations__ = {
        "box_coder": det_utils.BoxLinearCoder,
    }

    def __init__(self, in_channels: int, num_anchors: int, num_classes: int, num_convs: Optional[int] = 4) -> None:
        super().__init__()
        self.box_coder = det_utils.BoxLinearCoder(normalize_by_size=True)
        self.classification_head = FCOSClassificationHead(in_channels, num_anchors, num_classes, num_convs)
        self.regression_head = FCOSRegressionHead(in_channels, num_anchors, num_convs)

    def compute_loss(
        self,
        targets: List[Dict[str, Tensor]],
        head_outputs: Dict[str, Tensor],
        anchors: List[Tensor],
        matched_idxs: List[Tensor],
    ) -> Dict[str, Tensor]:

        cls_logits = head_outputs["cls_logits"]  # [N, HWA, C]
        bbox_regression = head_outputs["bbox_regression"]  # [N, HWA, 4]
        bbox_ctrness = head_outputs["bbox_ctrness"]  # [N, HWA, 1]

        dimensions_3d = head_outputs["dimensions_3d"]  # [N, HWA, 3]
        orientation = head_outputs["orientation"]  # [N, HWA, 1]
        keypoint = head_outputs["keypoint"] # [N, HWA, 16]
        depth = head_outputs["depth"]  # [N, HWA, 1]

        all_gt_classes_targets = []
        all_gt_boxes_targets = []
        all_gt_dimensions_targets = []  # For 3D dimensions
        all_gt_orientation_targets = [] # For orientation
        all_gt_keypoint_targets = []    # For location
        all_gt_depth_targets = []    # For location

        for targets_per_image, matched_idxs_per_image in zip(targets, matched_idxs):
            if len(targets_per_image["labels"]) == 0:
                gt_classes_targets = targets_per_image["labels"].new_zeros((len(matched_idxs_per_image),))
                gt_boxes_targets = targets_per_image["boxes"].new_zeros((len(matched_idxs_per_image), 4))
                gt_dimensions_targets = targets_per_image["dimensions_3d"].new_zeros((len(matched_idxs_per_image), 3))  # Assuming 3D dimensions
                gt_orientation_targets = targets_per_image["orientations_y"].new_zeros((len(matched_idxs_per_image), 1)) # Assuming quaternion or similar representation
                gt_keypoint_targets = targets_per_image["keypoint"].new_zeros((len(matched_idxs_per_image), 16))  # Assuming 3D location
                gt_depth_targets = targets_per_image["depth"].new_zeros((len(matched_idxs_per_image), 1))
            else:
                gt_classes_targets = targets_per_image["labels"][matched_idxs_per_image.clip(min=0)]
                gt_boxes_targets = targets_per_image["boxes"][matched_idxs_per_image.clip(min=0)]
                gt_dimensions_targets = targets_per_image["dimensions_3d"][matched_idxs_per_image.clip(min=0)]
                gt_orientation_targets = targets_per_image["orientations_y"][matched_idxs_per_image.clip(min=0)]
                gt_keypoint_targets = targets_per_image["keypoint"][matched_idxs_per_image.clip(min=0)]
                gt_depth_targets = targets_per_image["depth"][matched_idxs_per_image.clip(min=0)]

            gt_classes_targets[matched_idxs_per_image < 0] = -1  # background
            all_gt_classes_targets.append(gt_classes_targets)
            all_gt_boxes_targets.append(gt_boxes_targets)
            all_gt_dimensions_targets.append(gt_dimensions_targets)
            all_gt_orientation_targets.append(gt_orientation_targets)
            all_gt_keypoint_targets.append(gt_keypoint_targets)
            all_gt_depth_targets.append(gt_depth_targets)

        # List[Tensor] to Tensor conversion
        all_gt_boxes_targets, all_gt_classes_targets, all_gt_dimensions_targets, all_gt_orientation_targets, all_gt_keypoint_targets, all_gt_depth_targets, anchors = (
            torch.stack(all_gt_boxes_targets),
            torch.stack(all_gt_classes_targets),
            torch.stack(all_gt_dimensions_targets),
            torch.stack(all_gt_orientation_targets),
            torch.stack(all_gt_keypoint_targets),
            torch.stack(all_gt_depth_targets),
            torch.stack(anchors),
        )

        #print("all_gt_boxes_targets.shape", all_gt_boxes_targets.shape)
        #print("all_gt_classes_targets.shape", all_gt_classes_targets.shape)

        # compute foregroud
        foregroud_mask = all_gt_classes_targets >= 0
        num_foreground = foregroud_mask.sum().item()

        # classification loss
        gt_classes_targets = torch.zeros_like(cls_logits)
        gt_classes_targets[foregroud_mask, all_gt_classes_targets[foregroud_mask]] = 1.0
        loss_cls = sigmoid_focal_loss(cls_logits, gt_classes_targets, reduction="sum")

        # amp issue: pred_boxes need to convert float
        pred_boxes = self.box_coder.decode(bbox_regression, anchors)

        # regression loss: GIoU loss
        loss_bbox_reg = generalized_box_iou_loss(
            pred_boxes[foregroud_mask],
            all_gt_boxes_targets[foregroud_mask],
            reduction="sum",
        )

        # ctrness loss
        bbox_reg_targets = self.box_coder.encode(anchors, all_gt_boxes_targets)

        if len(bbox_reg_targets) == 0:
            gt_ctrness_targets = bbox_reg_targets.new_zeros(bbox_reg_targets.size()[:-1])
        else:
            left_right = bbox_reg_targets[:, :, [0, 2]]
            top_bottom = bbox_reg_targets[:, :, [1, 3]]
            epsilon = 1e-6  # A small constant to avoid division by zero

            # Add debugging print to check the values of left_right and top_bottom
            #print("Debug - left_right:", left_right.shape)
            #print("Debug - top_bottom:", top_bottom.shape)

            # Ensure non-negative values before sqrt operation
            left_right_min = left_right.min(dim=-1)[0].clamp(min=0)
            left_right_max = left_right.max(dim=-1)[0].clamp(min=epsilon)
            top_bottom_min = top_bottom.min(dim=-1)[0].clamp(min=0)
            top_bottom_max = top_bottom.max(dim=-1)[0].clamp(min=epsilon)

            gt_ctrness_targets = torch.sqrt(
                (left_right_min / left_right_max) * (top_bottom_min / top_bottom_max)
            )

            # Add debugging print to check the values of gt_ctrness_targets
            #print("Debug - gt_ctrness_targets shape:", gt_ctrness_targets.shape)

        pred_centerness = bbox_ctrness.squeeze(dim=2)

        # Add debugging print to check the values of pred_centerness
        #print("Debug - pred_centerness:", pred_centerness)

        loss_bbox_ctrness = nn.functional.binary_cross_entropy_with_logits(
            pred_centerness[foregroud_mask], gt_ctrness_targets[foregroud_mask], reduction="sum"
        )

        pred_dimensions_3d = dimensions_3d.squeeze(dim=2)
        pred_orientation = orientation.squeeze(dim=2)
        pred_keypoint = keypoint.squeeze(dim=2)

        loss_dimensions_3d = nn.functional.smooth_l1_loss(pred_dimensions_3d[foregroud_mask], all_gt_dimensions_targets[foregroud_mask], reduction="sum")/3
        loss_orientation = nn.functional.smooth_l1_loss(pred_orientation[foregroud_mask], all_gt_orientation_targets[foregroud_mask], reduction="sum")
        
        # Assuming you have a minimum depth to avoid taking log of zero
        min_depth = 1e-6

        # Apply logarithmic transformation to depth values
        log_gt_depth = torch.log(all_gt_depth_targets[foregroud_mask].clamp(min=min_depth))
        log_pred_depth = torch.log(depth[foregroud_mask].clamp(min=min_depth).squeeze(1))

        # Compute the depth loss for the predicted depths
        loss_depth = nn.functional.smooth_l1_loss(log_pred_depth, log_gt_depth, reduction="sum")

        # Apply logarithmic transformation to depth values
        log_gt_keypoint = torch.log(all_gt_keypoint_targets[foregroud_mask].clamp(min=min_depth))
        log_pred_keypoint = torch.log(pred_keypoint[foregroud_mask].clamp(min=min_depth).squeeze(1))
        loss_keypoint = nn.functional.smooth_l1_loss(log_pred_keypoint, log_gt_keypoint, reduction="sum")/16

        return {
            "classification": loss_cls / max(1, num_foreground),
            "bbox_regression": loss_bbox_reg / max(1, num_foreground),
            "bbox_ctrness": loss_bbox_ctrness / max(1, num_foreground),
            "dimensions_3d": loss_dimensions_3d / max(1, num_foreground),
            "orientation": loss_orientation / max(1, num_foreground),
            "keypoint": loss_keypoint / max(1, num_foreground),  # Include keypoint loss
            "depth": loss_depth / max(1, num_foreground)  # Include depth loss
        }

    def forward(self, x: List[Tensor]) -> Dict[str, Tensor]:
        cls_logits = self.classification_head(x)
        bbox_regression, bbox_ctrness, dimensions_3d, orientation, keypoint, depth = self.regression_head(x)
        #print("regression_head dimensions_3d", dimensions_3d)
        #print("regression_head bbox_regression", bbox_regression)
        return {
            "cls_logits": cls_logits,
            "bbox_regression": bbox_regression,
            "bbox_ctrness": bbox_ctrness,
            "dimensions_3d": dimensions_3d,
            "orientation": orientation,
            "keypoint": keypoint,
            "depth": depth
        }


class FCOSClassificationHead(nn.Module):
    """
    A classification head for use in FCOS.

    Args:
        in_channels (int): number of channels of the input feature.
        num_anchors (int): number of anchors to be predicted.
        num_classes (int): number of classes to be predicted.
        num_convs (Optional[int]): number of conv layer. Default: 4.
        prior_probability (Optional[float]): probability of prior. Default: 0.01.
        norm_layer: Module specifying the normalization layer to use.
    """

    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        num_classes: int,
        num_convs: int = 4,
        prior_probability: float = 0.01,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        if norm_layer is None:
            norm_layer = partial(nn.GroupNorm, 32)

        conv = []
        for _ in range(num_convs):
            conv.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(norm_layer(in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

        self.cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        torch.nn.init.constant_(self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability))

    def forward(self, x: List[Tensor]) -> Tensor:
        all_cls_logits = []

        for features in x:
            cls_logits = self.conv(features)
            cls_logits = self.cls_logits(cls_logits)

            # Permute classification output from (N, A * K, H, W) to (N, HWA, K).
            N, _, H, W = cls_logits.shape
            cls_logits = cls_logits.view(N, -1, self.num_classes, H, W)
            cls_logits = cls_logits.permute(0, 3, 4, 1, 2)
            cls_logits = cls_logits.reshape(N, -1, self.num_classes)  # Size=(N, HWA, 4)

            all_cls_logits.append(cls_logits)

        return torch.cat(all_cls_logits, dim=1)


class FCOSRegressionHead(nn.Module):
    """
    A regression head for use in FCOS, which combines regression branch and center-ness branch.
    This can obtain better performance.

    Reference: `FCOS: A simple and strong anchor-free object detector <https://arxiv.org/abs/2006.09214>`_.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_convs (Optional[int]): number of conv layer. Default: 4.
        norm_layer: Module specifying the normalization layer to use.
    """

    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        num_convs: int = 4,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()

        if norm_layer is None:
            norm_layer = partial(nn.GroupNorm, 32)

        conv = []
        for _ in range(num_convs):
            conv.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(norm_layer(in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        self.bbox_reg = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1)
        self.bbox_ctrness = nn.Conv2d(in_channels, num_anchors * 1, kernel_size=3, stride=1, padding=1)
        self.dimensions_3d_head = nn.Conv2d(in_channels, num_anchors * 3, kernel_size=3, stride=1, padding=1)
        self.orientation_head = nn.Conv2d(in_channels, num_anchors * 1, kernel_size=3, stride=1, padding=1)
        self.keypoint_head = nn.Conv2d(in_channels, num_anchors * 16, kernel_size=3, stride=1, padding=1)
        self.depth_head = nn.Conv2d(in_channels, num_anchors * 1, kernel_size=3, stride=1, padding=1)
        for layer in [self.bbox_ctrness, self.dimensions_3d_head, self.orientation_head, self.keypoint_head, self.depth_head]:
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.zeros_(layer.bias)

        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, x: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        all_bbox_regression = []
        all_bbox_ctrness = []
        all_dimensions_3d = []
        all_orientation = []
        all_keypoint = []
        all_depth = []

        for features in x:
            bbox_feature = self.conv(features)
            bbox_regression = nn.functional.relu(self.bbox_reg(bbox_feature))
            bbox_ctrness = self.bbox_ctrness(bbox_feature)


            # Predictions for 3D dimensions, orientation, and location
            dimensions_3d = self.dimensions_3d_head(bbox_feature)
            orientation = self.orientation_head(bbox_feature)
            keypoint = self.keypoint_head(bbox_feature)
            depth = self.depth_head(bbox_feature)

            # permute bbox regression output from (N, 4 * A, H, W) to (N, HWA, 4).
            N, _, H, W = bbox_regression.shape
            bbox_regression = bbox_regression.view(N, -1, 4, H, W)
            bbox_regression = bbox_regression.permute(0, 3, 4, 1, 2)
            bbox_regression = bbox_regression.reshape(N, -1, 4)  # Size=(N, HWA, 4)
            all_bbox_regression.append(bbox_regression)

            # permute bbox ctrness output from (N, 1 * A, H, W) to (N, HWA, 1).
            bbox_ctrness = bbox_ctrness.view(N, -1, 1, H, W)
            bbox_ctrness = bbox_ctrness.permute(0, 3, 4, 1, 2)
            bbox_ctrness = bbox_ctrness.reshape(N, -1, 1)
            all_bbox_ctrness.append(bbox_ctrness)

            # permute bbox ctrness output from (N, 3 * A, H, W) to (N, HWA, 3).
            dimensions_3d = dimensions_3d.view(N, -1, 3, H, W)
            dimensions_3d = dimensions_3d.permute(0, 3, 4, 1, 2)
            dimensions_3d = dimensions_3d.reshape(N, -1, 3)
            all_dimensions_3d.append(dimensions_3d)

            # permute bbox ctrness output from (N, 1 * A, H, W) to (N, HWA, 1).
            orientation = orientation.view(N, -1, 1, H, W)
            orientation = orientation.permute(0, 3, 4, 1, 2)
            orientation = orientation.reshape(N, -1, 1)
            all_orientation.append(orientation)

            # permute bbox ctrness output from (N, 3 * A, H, W) to (N, HWA, 3).
            keypoint = keypoint.view(N, -1, 16, H, W)
            keypoint = keypoint.permute(0, 3, 4, 1, 2)
            keypoint = keypoint.reshape(N, -1, 16)
            all_keypoint.append(keypoint)

            # permute bbox ctrness output from (N, 3 * A, H, W) to (N, HWA, 3).
            depth = depth.view(N, -1, 1, H, W)
            depth = depth.permute(0, 3, 4, 1, 2)
            depth = depth.reshape(N, -1, 1)
            all_depth.append(depth)

        return (
            torch.cat(all_bbox_regression, dim=1),
            torch.cat(all_bbox_ctrness, dim=1),
            torch.cat(all_dimensions_3d, dim=1),
            torch.cat(all_orientation, dim=1),
            torch.cat(all_keypoint, dim=1),
            torch.cat(all_depth, dim=1)
        )

class FCOS(nn.Module):
    """
    Implements FCOS.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

    The behavior of the model changes depending on if it is in training or evaluation mode.

    During training, the model expects both the input tensors and targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification, regression
    and centerness losses.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores for each prediction

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain an out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or an OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (including the background).
        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps. For FCOS, only set one anchor for per position of each level, the width and height equal to
            the stride of feature map, and set aspect ratio = 1.0, so the center of anchor is equivalent to the point
            in FCOS paper.
        head (nn.Module): Module run on top of the feature pyramid.
            Defaults to a module containing a classification and regression module.
        center_sampling_radius (int): radius of the "center" of a groundtruth box,
            within which all anchor points are labeled positive.
        score_thresh (float): Score threshold used for postprocessing the detections.
        nms_thresh (float): NMS threshold used for postprocessing the detections.
        detections_per_img (int): Number of best detections to keep after NMS.
        topk_candidates (int): Number of best detections to keep before NMS.

    Example:

        >>> import torch
        >>> import torchvision
        >>> from torchvision.models.detection import FCOS
        >>> from torchvision.models.detection.anchor_utils import AnchorGenerator
        >>> # load a pre-trained model for classification and return
        >>> # only the features
        >>> backbone = torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).features
        >>> # FCOS needs to know the number of
        >>> # output channels in a backbone. For mobilenet_v2, it's 1280,
        >>> # so we need to add it here
        >>> backbone.out_channels = 1280
        >>>
        >>> # let's make the network generate 5 x 3 anchors per spatial
        >>> # location, with 5 different sizes and 3 different aspect
        >>> # ratios. We have a Tuple[Tuple[int]] because each feature
        >>> # map could potentially have different sizes and
        >>> # aspect ratios
        >>> anchor_generator = AnchorGenerator(
        >>>     sizes=((8,), (16,), (32,), (64,), (128,)),
        >>>     aspect_ratios=((1.0,),)
        >>> )
        >>>
        >>> # put the pieces together inside a FCOS model
        >>> model = FCOS(
        >>>     backbone,
        >>>     num_classes=80,
        >>>     anchor_generator=anchor_generator,
        >>> )
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
    """

    __annotations__ = {
        "box_coder": det_utils.BoxLinearCoder,
    }

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        # transform parameters
        min_size: int = 800,
        max_size: int = 1333,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        # Anchor parameters
        anchor_generator: Optional[AnchorGenerator] = None,
        head: Optional[nn.Module] = None,
        center_sampling_radius: float = 1.5,
        score_thresh: float = 0.5,
        nms_thresh: float = 0.3,
        detections_per_img: int = 100,
        topk_candidates: int = 1000,
        **kwargs,
    ):
        super().__init__()
        #_log_api_usage_once(self)

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)"
            )
        self.backbone = backbone

        if not isinstance(anchor_generator, (AnchorGenerator, type(None))):
            raise TypeError(
                f"anchor_generator should be of type AnchorGenerator or None, instead  got {type(anchor_generator)}"
            )

        if anchor_generator is None:
            anchor_sizes = ((8,), (16,), (32,), (64,), (128,))  # equal to strides of multi-level feature map
            aspect_ratios = ((1.0,),) * len(anchor_sizes)  # set only one anchor
            anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        self.anchor_generator = anchor_generator
        if self.anchor_generator.num_anchors_per_location()[0] != 1:
            raise ValueError(
                f"anchor_generator.num_anchors_per_location()[0] should be 1 instead of {anchor_generator.num_anchors_per_location()[0]}"
            )

        if head is None:
            head = FCOSHead(backbone.out_channels, anchor_generator.num_anchors_per_location()[0], num_classes)
        self.head = head

        self.box_coder = det_utils.BoxLinearCoder(normalize_by_size=True)

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std, **kwargs)

        self.center_sampling_radius = center_sampling_radius
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.topk_candidates = topk_candidates

        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(
        self, losses: Dict[str, Tensor], detections: List[Dict[str, Tensor]]
    ) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        if self.training:
            return losses

        return detections

    def compute_loss(
        self,
        targets: List[Dict[str, Tensor]],
        head_outputs: Dict[str, Tensor],
        anchors: List[Tensor],
        num_anchors_per_level: List[int],
    ) -> Dict[str, Tensor]:
        matched_idxs = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            if targets_per_image["boxes"].numel() == 0:
                matched_idxs.append(
                    torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device)
                )
                continue

            gt_boxes = targets_per_image["boxes"]
            gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2  # Nx2
            anchor_centers = (anchors_per_image[:, :2] + anchors_per_image[:, 2:]) / 2  # N
            anchor_sizes = anchors_per_image[:, 2] - anchors_per_image[:, 0]
            # center sampling: anchor point must be close enough to gt center.
            pairwise_match = (anchor_centers[:, None, :] - gt_centers[None, :, :]).abs_().max(
                dim=2
            ).values < self.center_sampling_radius * anchor_sizes[:, None]
            # compute pairwise distance between N points and M boxes
            x, y = anchor_centers.unsqueeze(dim=2).unbind(dim=1)  # (N, 1)
            x0, y0, x1, y1 = gt_boxes.unsqueeze(dim=0).unbind(dim=2)  # (1, M)
            pairwise_dist = torch.stack([x - x0, y - y0, x1 - x, y1 - y], dim=2)  # (N, M)

            # anchor point must be inside gt
            pairwise_match &= pairwise_dist.min(dim=2).values > 0

            # each anchor is only responsible for certain scale range.
            lower_bound = anchor_sizes * 4
            lower_bound[: num_anchors_per_level[0]] = 0
            upper_bound = anchor_sizes * 8
            upper_bound[-num_anchors_per_level[-1] :] = float("inf")
            pairwise_dist = pairwise_dist.max(dim=2).values
            pairwise_match &= (pairwise_dist > lower_bound[:, None]) & (pairwise_dist < upper_bound[:, None])

            # match the GT box with minimum area, if there are multiple GT matches
            gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])  # N
            pairwise_match = pairwise_match.to(torch.float32) * (1e8 - gt_areas[None, :])
            min_values, matched_idx = pairwise_match.max(dim=1)  # R, per-anchor match
            matched_idx[min_values < 1e-5] = -1  # unmatched anchors are assigned -1

            matched_idxs.append(matched_idx)

        return self.head.compute_loss(targets, head_outputs, anchors, matched_idxs)

    def postprocess_detections(
    self, head_outputs: Dict[str, List[Tensor]], anchors: List[List[Tensor]], image_shapes: List[Tuple[int, int]]
) -> List[Dict[str, Tensor]]:
        class_logits = head_outputs["cls_logits"]
        box_regression = head_outputs["bbox_regression"]
        box_ctrness = head_outputs["bbox_ctrness"]
        dimensions_3d = head_outputs["dimensions_3d"]  # List of Tensors
        orientation = head_outputs["orientation"]  # List of Tensors
        keypoint = head_outputs["keypoint"]  # List of Tensors
        depth = head_outputs["depth"]  # List of Tensors

        num_images = len(image_shapes)
        detections = [{} for _ in range(num_images)]  # Initialize detections for each image


        #detections: List[Dict[str, Tensor]] = []

        for index in range(num_images):
            box_regression_per_image = [br[index] for br in box_regression]
            logits_per_image = [cl[index] for cl in class_logits]
            box_ctrness_per_image = [bc[index] for bc in box_ctrness]
            anchors_per_image, image_shape = anchors[index], image_shapes[index]

            dimensions_3d_per_image = [dim[index] for dim in dimensions_3d] if dimensions_3d else None
            orientation_per_image = [rot[index] for rot in orientation] if orientation else None
            keypoint_per_image = [kpt[index] for kpt in keypoint] if keypoint else None
            depth_per_image = [dep[index] for dep in depth] if depth else None

            image_boxes = []
            image_scores = []
            image_labels = []
            image_dimensions_3d = []
            image_orientation = []
            image_keypoint = []
            image_depth = []

            for box_regression_per_level, logits_per_level, box_ctrness_per_level, anchors_per_level, dimensions_3d_per_level, orientation_per_level, keypoint_per_level, depth_per_level in zip(
                box_regression_per_image, logits_per_image, box_ctrness_per_image, anchors_per_image, dimensions_3d_per_image, orientation_per_image, keypoint_per_image, depth_per_image
            ):
                num_classes = logits_per_level.shape[-1]

                # remove low scoring boxes
                scores_per_level = torch.sqrt(
                    torch.sigmoid(logits_per_level) * torch.sigmoid(box_ctrness_per_level)
                ).flatten()
                keep_idxs = scores_per_level > self.score_thresh
                scores_per_level = scores_per_level[keep_idxs]
                topk_idxs = torch.where(keep_idxs)[0]

                # keep only topk scoring predictions
                num_topk = det_utils._topk_min(topk_idxs, self.topk_candidates, 0)
                scores_per_level, idxs = scores_per_level.topk(num_topk)
                topk_idxs = topk_idxs[idxs]

                anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
                labels_per_level = topk_idxs % num_classes

                boxes_per_level = self.box_coder.decode(
                    box_regression_per_level[anchor_idxs], anchors_per_level[anchor_idxs]
                )
                boxes_per_level = box_ops.clip_boxes_to_image(boxes_per_level, image_shape)

                image_boxes.append(boxes_per_level)
                image_scores.append(scores_per_level)
                image_labels.append(labels_per_level)


                # Decode 3D outputs for the current level
                dimensions_3d_level = dimensions_3d_per_level[anchor_idxs]
                orientation_level = orientation_per_level[anchor_idxs]
                keypoint_level = keypoint_per_level[anchor_idxs]
                depth_level = depth_per_level[anchor_idxs]
                # Append all outputs to the respective lists
                image_dimensions_3d.append(dimensions_3d_level)
                image_keypoint.append(keypoint_level) 
                image_orientation.append(orientation_level)
                image_depth.append(depth_level)

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)
            # Concatenate all outputs for the current image
            image_dimensions_3d = torch.cat(image_dimensions_3d, dim=0)
            image_orientation = torch.cat(image_orientation, dim=0)
            image_keypoint = torch.cat(image_keypoint, dim=0)
            image_depth = torch.cat(image_depth, dim=0)

            # Filter based on NMS keep indices
            keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
            keep = keep[: self.detections_per_img]
            #image_boxes = image_boxes[keep]
            #image_scores = image_scores[keep]
            #image_labels = image_labels[keep]
            #image_dimensions_3d = image_dimensions_3d[keep]
            #image_locations_3d = image_locations_3d[keep]
            #image_orientation = image_orientation[keep]


            detections[index]["boxes"] = image_boxes[keep]
            detections[index]["scores"] = image_scores[keep]
            detections[index]["labels"] = image_labels[keep]

            detections[index]["dimensions_3d"] = image_dimensions_3d[keep]
            detections[index]["orientation"] = image_orientation[keep]
            detections[index]["keypoint"] = image_keypoint[keep]
            detections[index]["depth"] = image_depth[keep]



        return detections


    def forward(
        self,
        images: List[Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        if self.training:

            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    torch._assert(isinstance(boxes, torch.Tensor), "Expected target boxes to be of type Tensor.")
                    torch._assert(
                        len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                        f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                    )

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))# Before transformation
        
        #print("Before Transform:")
        #print("Images shapes: ", [image.shape for image in images])
        #print("Targets shapes: ", [target['boxes'].shape for target in targets])

        # transform the input
        images, targets = self.transform(images, targets)

        # After transformation
        #print("After Transform:")
        ## The 'images' after transform is a transformed batch, not a list, hence direct .shape can be used
        #print("Images tensor shape: ", images.tensors.shape)
        #print("Targets shapes: ", [target['boxes'].shape for target in targets])


        # Check for degenerate boxes
        if targets is not None:
            for target_idx, target in enumerate(targets):
                #print("FCOS_forward: target", target)
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        f"All bounding boxes should have positive height and width. Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        # get the features from the backbone
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        features = list(features.values())

        # compute the fcos heads outputs using the features
        head_outputs = self.head(features)

        # create the set of anchors
        anchors = self.anchor_generator(images, features)
        # recover level sizes
        num_anchors_per_level = [x.size(2) * x.size(3) for x in features]

        losses = {}
        detections: List[Dict[str, Tensor]] = []
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                # compute the losses
                losses = self.compute_loss(targets, head_outputs, anchors, num_anchors_per_level)
        else:
            # split outputs per level
            split_head_outputs: Dict[str, List[Tensor]] = {}
            for k in head_outputs:
                split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))
            split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

            # compute the detections
            detections = self.postprocess_detections(split_head_outputs, split_anchors, images.image_sizes)
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        #print("forward detections", detections)
        #print("self.eager_outputs(losses, detections)", self.eager_outputs(losses, detections))
        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("FCOS always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        return self.eager_outputs(losses, detections)

def _overwrite_value_param(params: Dict[str, Any], key: str, value: Any) -> None:
    """
    Overwrite a parameter value in a dictionary if the key exists.

    Args:
        params (Dict[str, Any]): Dictionary of parameters.
        key (str): The key of the parameter to be overwritten.
        value (Any): The new value to set for the parameter.
    """
    if key in params and value is not None:
        params[key] = value

def fcos3d(
    *,
    progress: bool = True,
    num_classes: Optional[int] = None,
    weights_backbone: Optional[ResNet101_Weights] = ResNet101_Weights.IMAGENET1K_V1,
    trainable_backbone_layers: Optional[int] = None,
    **kwargs: Any,
) -> FCOS:
    # Verify and load the backbone weights
    if weights_backbone is not None:
        weights_backbone = ResNet101_Weights.verify(weights_backbone)
        backbone = resnet101(weights=weights_backbone, progress=progress)
    else:
        # Initialize the backbone without pretrained weights
        backbone = resnet101(weights=None, progress=progress)

    # Extract FPN from the backbone
    trainable_backbone_layers = _validate_trainable_layers(weights_backbone is not None, trainable_backbone_layers, 5, 3)
    norm_layer = misc_nn_ops.FrozenBatchNorm2d if weights_backbone is not None else nn.BatchNorm2d
    backbone = _resnet_fpn_extractor(
        backbone, trainable_backbone_layers, returned_layers=[2, 3, 4], extra_blocks=LastLevelP6P7(256, 256)
    )

    # Rest of your FCOS model initialization
    model = FCOS(backbone, num_classes, **kwargs)
    return model

