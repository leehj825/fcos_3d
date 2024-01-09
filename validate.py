import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.optim as optim
import argparse
from PIL import Image
from tqdm.auto import tqdm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

import dataset.kitti as kitti
import dataset.waymo as waymo
import model.fcos3d as fcos3d
import metric.metric_3d as metric


from torchvision.models.detection import fcos
from torchvision.datasets import VOCDetection
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import torch.nn.functional as Func
import torchvision.ops as ops

from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from PIL import Image, ImageDraw, ImageOps, ImageFont
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchvision.models.resnet import ResNet101_Weights

import numpy as np

from collections import defaultdict


# Set environment variable for MPS (if using Apple Silicon)
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# Define class mapping for the KITTI dataset
CLASS_MAPPING = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}

# Default paths and parameters for KITTI dataset
default_kitti_data_path = "/Users/hyejunlee/fcos_3d/data/kitti_200/"
default_kitti_image_path = '/Users/hyejunlee/fcos_3d/data/kitti_200/training/image_2/000025.png'
default_kitti_label_folder = '/Users/hyejunlee/fcos_3d/data/kitti_200/training/label_2/'
default_kitti_calib_folder = '/Users/hyejunlee/fcos_3d/data/kitti_200/training/calib/'

# Default paths and parameters for Waymo dataset
default_waymo_data_path = "/Users/hyejunlee/fcos_3d/data/waymo_single/"  # Update this path as per your Waymo dataset location
default_waymo_image_path = '/Users/hyejunlee/fcos_3d/data/waymo_single/training/image_0/0000001.jpg'  # Update with a Waymo image path
default_waymo_label_folder = '/Users/hyejunlee/fcos_3d/data/waymo_single/training/label_0/'
default_waymo_calib_folder = '/Users/hyejunlee/fcos_3d/data/waymo_single/training/calib/'
# Add more Waymo specific paths and parameters if needed

default_learning_rate = 0.001

default_image_path ='data/kitti_200/training/image_2/000005.png'
#default_load_checkpoint = '/Users/hyejunlee/fcos_3d/save_state_kitti_20.bin'
default_load_checkpoint = '/Users/hyejunlee/fcos_3d/save_state_waymo_20.bin'
#default_load_checkpoint = None

#default_output_image_path = 'output_kitti_20'
default_output_image_path = 'save_state_waymo_20_val'
num_images = 5

# Check if the directory exists
if not os.path.exists(default_output_image_path):
    os.makedirs(default_output_image_path)  # Create the directory if it does not exist

# Detect device
#device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
device = "cpu"
print(device)


def custom_collate(batch, dataset_name):
    # Check each data entry in the batch
    for data in batch:
        images, targets, calib_data, _ = data
        # If any of images, targets, or calib_data is None, return None for the entire batch
        if images is None or targets is None or calib_data is None:
            return None
    
    # Define resize transformations for KITTI and Waymo datasets
    resize_transform_kitti = transforms.Compose([
        transforms.Resize((375, 1242)),
        transforms.ToTensor()
    ])
    resize_transform_waymo = transforms.Compose([
        transforms.Resize((1280, 1920)),  # Example resize for Waymo; adjust as needed
        transforms.ToTensor()
    ])

    images, targets, calib_data, image_path = zip(*batch)
    
    # Apply the appropriate resize transformation based on the dataset
    if dataset_name == 'kitti':
        images = [resize_transform_kitti(img) for img in images]
    elif dataset_name == 'waymo':
        images = [resize_transform_waymo(img) for img in images]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    images = torch.stack(images, 0).to(device)

    # Calibration data handling might differ between datasets
    # ...

    return images, targets, calib_data, image_path

def preprocess_image(img_path, dataset_name):
    image = Image.open(img_path)

    # Apply the appropriate resize transformation based on the dataset
    if dataset_name == 'kitti':
        resize_transform = transforms.Compose([
            transforms.Resize((375, 1242)),
            transforms.ToTensor()
        ])
    elif dataset_name == 'waymo':
        resize_transform = transforms.Compose([
            transforms.Resize((1280, 1920)),  # Example resize for Waymo; adjust as needed
            transforms.ToTensor()
        ])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    image = resize_transform(image).unsqueeze(0).to(device)
    return image

def inference(model, image_path, dataset_name):
    # Perform inference on a given image
    model.eval()
    with torch.no_grad():
        image = preprocess_image(image_path, dataset_name)
        predictions = model(image)
        # Example post-processing (adjust as per your model's output format)
        # Assuming predictions is a list of dictionaries with 'boxes', 'scores', 'labels', 'dimensions', 'locations', 'rotations' keys
        if predictions:
            pred = predictions[0]  # Assuming batch size of 1
            boxes = pred['boxes']
            scores = pred['scores']
            labels = pred['labels']
            dimensions = pred['dimensions_3d']
            locations = pred['location_xy']
            orientation = pred['orientation']
            depth = pred['depth']
        else:
            boxes, scores, labels, dimensions, locations, orientation, depth = [], [], [], [], [], [], []
    return boxes, scores, labels, dimensions, locations, orientation, depth


def project_to_image(pts_3d, P):
    """ Project 3D points to 2D by applying the calibration matrix P """
    pts_3d_hom = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)))
    pts_2d_hom = np.dot(pts_3d_hom, np.transpose(P))  # Project
    pts_2d = pts_2d_hom[:, :2] / pts_2d_hom[:, 2:3]  # Normalize
    return pts_2d

def draw_3d_box(draw, corners_2d, color="red"):
    """ Draw 3D bounding box given 2D corners """
    # Define connections between corners
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    # Draw each edge
    for i, j in edges:
        draw.line([tuple(corners_2d[i]), tuple(corners_2d[j])], fill=color, width=5)


def create_3d_bbox(dimensions, location, rotation_y):
    # Dimensions of the bounding box
    
    #print("Dimensions:", dimensions, "Type:", type(dimensions))
    #print("Location:", location, "Type:", type(location))
    #print("Rotation Y:", rotation_y, "Type:", type(rotation_y))

    # Convert PyTorch tensors to NumPy arrays
    dimensions = dimensions.cpu().numpy() if isinstance(dimensions, torch.Tensor) else dimensions
    location = location.cpu().numpy() if isinstance(location, torch.Tensor) else location
    rotation_y = rotation_y.item() if isinstance(rotation_y, torch.Tensor) else rotation_y

    h, w, l = dimensions
    # Location of the bounding box
    x, y, z = location

    # Rotation matrix around the Y-axis
    R = np.array([
        [np.cos(rotation_y), 0, np.sin(rotation_y)],
        [0, 1, 0],
        [-np.sin(rotation_y), 0, np.cos(rotation_y)]
    ])

    # 3D bounding box corners
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

    # Rotate and translate 3D bounding box corners
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] += x
    corners_3d[1, :] += y
    corners_3d[2, :] += z
    return corners_3d

def back_project_to_3d(x_2d, y_2d, z_depth, P):
    # Inverting the intrinsic matrix
    K = P[:, :3]

    # Create the normalized image coordinates
    x_normalized = (x_2d - K[0, 2]) / K[0, 0]
    y_normalized = (y_2d - K[1, 2]) / K[1, 1]

    # Scale by depth to get the 3D point
    x_3d = x_normalized * z_depth
    y_3d = y_normalized * z_depth

    return np.array([x_3d, y_3d, z_depth])


def create_3d_bbox_2d(dimensions, box_2d, location_z, rotation_y, P):
    # Dimensions of the bounding box
    dimensions = dimensions.cpu().numpy() if isinstance(dimensions, torch.Tensor) else dimensions
    rotation_y = rotation_y.item() if isinstance(rotation_y, torch.Tensor) else rotation_y

    h, w, l = dimensions

    # Use the center of the 2D bounding box as x, y
    x_center_2d = (box_2d[0] + box_2d[2]) / 2
    y_center_2d = (box_2d[1] + box_2d[3]) / 2

    #print("location_z", location_z)
    # Use the z-coordinate from the 3D location
    z = location_z.item() if isinstance(location_z, torch.Tensor) else location_z

    # Rotation matrix around the Y-axis
    R = np.array([
        [np.cos(rotation_y), 0, np.sin(rotation_y)],
        [0, 1, 0],
        [-np.sin(rotation_y), 0, np.cos(rotation_y)]
    ])
    
    # Assuming you have access to the intrinsic matrix K for your camera
    x_center_3d, y_center_3d, z_depth= back_project_to_3d(x_center_2d, y_center_2d, z, P)

    # 3D bounding box corners
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    #y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    y_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

    # Rotate and translate 3D bounding box corners
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] += x_center_3d
    corners_3d[1, :] += y_center_3d
    corners_3d[2, :] += z_depth

    return corners_3d


# Rest of the functions (project_to_image, draw_3d_box) remain the same as in the previous example

def save_combined_image(dataset_name, calib_data, boxes, scores, labels, dimensions, locations, orientation, image_path, output_image_path):
    
    if dataset_name == 'kitti':
        #default_calib_folder = default_kitti_calib_folder
        default_label_folder = default_kitti_label_folder
    elif dataset_name == 'waymo':
        #default_calib_folder = default_waymo_calib_folder
        default_label_folder = default_waymo_label_folder
    
    # Open the original image
    image_gt = Image.open(image_path)
    draw_gt = ImageDraw.Draw(image_gt)

    #image_pred = Image.open(image_path)
    #draw_pred = ImageDraw.Draw(image_pred)

    # Get label file path
    #calib_file = os.path.splitext(os.path.basename(image_path))[0] + '.txt'
    #calib_path = os.path.join(default_calib_folder, calib_file)

    # Get label file path
    label_file = os.path.splitext(os.path.basename(image_path))[0] + '.txt'
    label_path = os.path.join(default_label_folder, label_file)


    # Read calibration data
    #calib_data = {}
    #with open(calib_path, 'r') as calib_file:
    #    for line in calib_file:
    #        if ':' in line:  # Ensure line contains a colon
    #            key, value = line.split(':', 1)
    #            calib_data[key] = np.fromstring(value, sep=' ')

    # Extract calibration matrix
    #if dataset_name == 'kitti':
    #    P = calib_data.get('P2').reshape(3, 4) if 'P2' in calib_data else None
    #elif dataset_name == 'waymo':
    #    P = calib_data.get('P0').reshape(3, 4) if 'P0' in calib_data else None
    P = calib_data
    if P is None:
        raise ValueError("P2 (kitti) or P0 (waymo) calibration matrix not found in calibration data.")

    # Draw ground truth bounding boxes in yellow and 3D bounding boxes
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            bbox = [float(parts[i]) for i in range(4, 8)]  # 2D bounding box

            dimension_3d = [float(parts[i]) for i in range(8, 11)]  # 3D dimensions
            location_3d = [float(parts[i]) for i in range(11, 14)]  # 3D location
            rotation_y = float(parts[14])  # Orientation

            corners_3d = create_3d_bbox(dimension_3d, location_3d, rotation_y)
            corners_2d = project_to_image(corners_3d.T, P)

            #draw.rectangle(bbox, outline="yellow")
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:  # Ensures width and height are positive
                draw_3d_box(draw_gt, corners_2d)

    # Draw prediction bounding boxes in green
    #for box, score, label in zip(boxes, scores, labels):
    #print("dimensions out of loop", dimensions)
    for box, score, label, dimension, location, orientation, depth in zip(boxes, scores, labels, dimensions, locations, orientation, depth):
        if score > 0.5:  # Threshold can be adjusted
            #draw_pred.rectangle(box.tolist(), outline="blue")  # Draw 2D bounding box in blue
            #print("dimension in the loop", dimension)
            
            # Draw 3D box
            #corners_3d = create_3d_bbox(dimension, location, orientation)

            #corners_2d = project_to_image(corners_3d.T, P)
            #print("corners_2d", corners_2d)
            #draw_3d_box(draw_gt, corners_2d, color="blue")

            corners_3d = create_3d_bbox_2d(dimension, box, location[2], orientation, P)
            #print("bbox", box)
            #print("corners_3d", corners_3d)
            corners_2d = project_to_image(corners_3d.T, P)
            #print("corners_2d", corners_2d)
            #draw_3d_box(draw_pred, corners_2d, color="green")
            draw_3d_box(draw_gt, corners_2d, color="green")
    
    # Save the combined image
    #image.save(output_image_path)

    # Combine the two images (stack them vertically)
    #combined_image = Image.new('RGB', (image_gt.width, image_gt.height * 2))
    combined_image = Image.new('RGB', (image_gt.width, image_gt.height))
    combined_image.paste(image_gt, (0, 0))
    #combined_image.paste(image_pred, (0, image_gt.height))

    # Save the combined image
    combined_image.save(output_image_path)


def main(mode='inference', dataset_name='waymo', image_path=None, load=None):
    # Main function to handle training and inference
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    if mode == 'inference':
        device = "cpu"

    print(device)

    # Load the pre-trained FCOS model
    model = fcos3d.fcos3d(weights_backbone=ResNet101_Weights.IMAGENET1K_V1, num_classes=3)
    model = model.to(device)

    # Construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=default_learning_rate, weight_decay=0.0005)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=1, verbose=True)

    model.zero_grad()

    # Load the appropriate dataset
    if dataset_name == 'kitti':
        dataset = kitti.Kitti(root=default_kitti_data_path, train=True)
    elif dataset_name == 'waymo':
        dataset = waymo.Waymo(root=default_waymo_data_path, train=True)  # Ensure this is correctly implemented
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    

    # Define a data loader
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda batch: custom_collate(batch, dataset_name))
    #data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)

    #num_epochs = 500
    #start_epoch = 0

    # Load model checkpoint if provided
    #if load:
    #    checkpoint = torch.load(load)
    #    start_epoch = checkpoint['epoch']
    #    model.load_state_dict(checkpoint['model_state_dict'])
    #    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if load:
        # Loading checkpoint for resuming training
        # Try loading the checkpoint
        try:

            print(f"Loading checkpoint: {load}")
            checkpoint = torch.load(load)
            model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            print(f"Error loading checkpoint: {e}")

            # Attempt to load with map_location
            try:
                checkpoint = torch.load(load, map_location=torch.device('cpu'))
                model.load_state_dict(checkpoint['model_state_dict'])
            except RuntimeError as e:
                print(f"Error loading checkpoint with map_location: {e}")

                # Attempt to load with specific pickle_module and encoding
                import pickle
                try:
                    checkpoint = torch.load(load, pickle_module=pickle, encoding='utf-8')
                    model.load_state_dict(checkpoint['model_state_dict'])
                except RuntimeError as e:
                    print(f"Error loading checkpoint with pickle_module: {e}")
                    # If this fails, the file might be corrupted or not a valid checkpoint
            
    if mode == 'inference':
        device = "cpu"
        model = model.to(device)

        processed_images = 0
        iou_thresholds = [0.0, 0.1, 0.2, 0.3, 0.4]  # Define the IoU thresholds you want to use

        for batch_idx, (images, targets, calib_data, image_paths) in enumerate(data_loader):
            if num_images and processed_images >= num_images:
                break  # Stop if we have processed the desired number of images

            images = images.to(device)

            original_image_path = image_paths[0]
            
            for idx, image in enumerate(images):
                image_path = image_paths[idx]
                single_image_predictions = []
                single_image_ground_truths = []

                # Perform inference on single image
                detections = inference(model, image_path, dataset_name)
                boxes, scores, labels, dimensions, locations, rotations, depth = detections

                print("detections", detections)
                #print("orientation", detections)
                original_image_path = image_paths[0]
                save_combined_image(dataset_name, calib_data[idx], boxes, scores, labels, dimensions, locations, rotations, original_image_path, f"{default_output_image_path}/output_{batch_idx}.png")

                processed_images += 1

                # Convert detections to the required format
                for i in range(len(boxes)):
                    box_2d_center_x = (boxes[i][0] + boxes[i][2]) / 2
                    box_2d_center_y = (boxes[i][1] + boxes[i][3]) / 2
                    location_3d = back_project_to_3d(box_2d_center_x, box_2d_center_y, locations[i][-1], calib_data[idx][:, :3])  # Use z component from locations[i]
                    #print("location_3d", location_3d)
                    # Convert to list and ensure double precision
                    location_3d = [float(coord) for coord in location_3d.tolist()] if isinstance(location_3d, (torch.Tensor, np.ndarray)) else location_3d
                    dimension_3d = [float(dim) for dim in dimensions[i].tolist()] if isinstance(dimensions[i], (torch.Tensor, np.ndarray)) else dimensions[i]
                    orientation = float(rotations[i].item()) if isinstance(rotations[i], torch.Tensor) else rotations[i]

                    pred_box = {
                        'location_3d': location_3d,
                        'dimension_3d': dimension_3d,
                        'orientation': orientation
                    }
                    #print("pred_box", i , pred_box)
                    single_image_predictions.append(pred_box)

                # Extract ground truth data for single image
                image_targets = targets[idx]
                for target in image_targets:
                    if target["type"] in CLASS_MAPPING:
                        if target["bbox"][2] > target["bbox"][0] and target["bbox"][3] > target["bbox"][1]:  # Ensures width and height are positive
                            #print("target['location']", target['location'])
                            gt_box = {
                                'location_3d': target['location'],
                                'dimension_3d': target['dimensions'],
                                'orientation': target['rotation_y']
                            }
                            single_image_ground_truths.append(gt_box)

                # Compute mAP for single image
                #map_value = metric.compute_map(single_image_predictions, single_image_ground_truths, iou_thresholds)
                #print(f"Image: {image_path}, mAP: {map_value}")
                #print("single_image_predictions", single_image_predictions)
                #print("single_image_ground_truths", single_image_ground_truths)

                # Compute mAP for single image
                map_values_by_threshold = metric.compute_map(single_image_predictions, single_image_ground_truths, iou_thresholds)

                # Print mAP for each threshold
                print(f"Image: {image_path}")
                print(f"  mAP: {map_values_by_threshold}\n\n")

        print(f"Completed inference on {processed_images} images.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FCOS Training/Inference')
    parser.add_argument('--mode', type=str, default='inference', choices=['train', 'inference'], help='Mode to run the script in')
    parser.add_argument('--dataset', type=str, default='waymo', choices=['kitti', 'waymo'], help='Dataset to use')
    parser.add_argument('--image_path', default=default_kitti_image_path, type=str, help='Path to the image for inference mode')
    parser.add_argument('--load', default=default_load_checkpoint, type=str, help='Path to the pretrained model file')
    args = parser.parse_args()
    main(mode=args.mode, dataset_name=args.dataset, image_path=args.image_path, load=args.load)
