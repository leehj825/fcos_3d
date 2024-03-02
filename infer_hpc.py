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
import metric.metric as metric


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
default_kitti_data_path = "/scratch/cmpe249-fa23/kitti_od/"
default_kitti_image_path = '/scratch/cmpe249-fa23/kitti_od/training/image_2/000025.png'
default_kitti_label_folder = '/scratch/cmpe249-fa23/kitti_od/training/label_2/'
default_kitti_calib_folder = '/scratch/cmpe249-fa23/kitti_od/training/calib/'

# Default paths and parameters for Waymo dataset
default_waymo_data_path = "/scratch/cmpe249-fa23/waymo_data/waymo_single/kitti_format/"  # Update this path as per your Waymo dataset location
default_waymo_image_path = '/scratch/cmpe249-fa23/waymo_data/waymo_single/kitti_format/training/image_0/0000001.jpg'  # Update with a Waymo image path
default_waymo_label_folder = '/scratch/cmpe249-fa23/waymo_data/waymo_single/kitti_format/training/label_0/'
default_waymo_calib_folder = '/scratch/cmpe249-fa23/waymo_data/waymo_single/kitti_format/training/calib/'
# Add more Waymo specific paths and parameters if needed

default_learning_rate = 0.001

default_image_path ='data/kitti_200/training/image_2/000005.png'
#default_load_checkpoint = '/home/001891254/fcos_3d/save_state_kitti_sim_depth_hpc_290.bin'
default_load_checkpoint = '/home/001891254/fcos_3d_new_depth/save_state_waymo_new_depth_30.bin'
#default_load_checkpoint = 'save_state_waymo_40.bin'
#default_load_checkpoint = None

#default_output_image_path = 'output_kitti_sim_depth_hpc_290'
default_output_image_path = 'output_waymo_hpc_30'
#default_output_image_path = 'output_waymo_30'
num_images = 2

# Check if the directory exists
if not os.path.exists(default_output_image_path):
    os.makedirs(default_output_image_path)  # Create the directory if it does not exist

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
device = "cpu"
print(device)

def custom_collate(batch, dataset_name):
    # Process images and handle cases where targets or calib_data may not be present
    images = []
    image_paths = []
    calib_data = []
    for data in batch:
        images.append(data[0])  # Always add the image
        calib_data.append(data[2])
        image_paths.append(data[-1])  # The image path is assumed to be the last element

    # Define resize transformations for KITTI and Waymo datasets
    resize_transform_kitti = transforms.Compose([
        transforms.Resize((375, 1242)),
        transforms.ToTensor()
    ])
    resize_transform_waymo = transforms.Compose([
        transforms.Resize((1280, 1920)),  # Example resize for Waymo; adjust as needed
        transforms.ToTensor()
    ])
    
    # Apply the appropriate resize transformation based on the dataset
    if dataset_name == 'kitti':
        images = [resize_transform_kitti(img) for img in images]
    elif dataset_name == 'waymo':
        images = [resize_transform_waymo(img) for img in images]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    images = torch.stack(images, 0).to(device)

    return images, None, calib_data, image_paths

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
            #orientation = pred['orientation']
            # Extract sine and cosine values for orientation
            orientation_sin = torch.sin(pred['orientation'][:, 0])  # Assuming the first channel is sine
            orientation_cos = torch.cos(pred['orientation'][:, 1])  # Assuming the second channel is cosine
            # Optionally convert sine and cosine back to angle
            orientation_angle = torch.atan2(orientation_sin, orientation_cos)  # Resulting angle in radians

            depth = pred['depth']
        else:
            boxes, scores, labels, dimensions, orientation_angle, depth = [], [], [], [], [], [], []
    return boxes, scores, labels, dimensions, orientation_angle, depth


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
    #print("Type of P in back_project_to_3d:", type(P))
    #print("Shape of P in back_project_to_3d:", P.shape)
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


def generate_gif(folder_path, output_gif_path):
    # Folder containing PNG files
    #folder_path = '/Users/hyejunlee/fcos_3d/output_waymo_sim_depth_hpc_30'

    # Output GIF file name
    #output_gif_path = 'output_waymo_sim_depth_hpc_30.gif'

    # Read all PNG files in folder
    png_files = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path)) if f.endswith('.png')]

    # Load images
    images = [Image.open(png) for png in png_files]

    # Save the images as a GIF
    images[0].save(output_gif_path, save_all=True, append_images=images[1:], optimize=False, duration=200, loop=0)


def get_color_for_label(label):
    # List of colors
    colors = ['blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'gray', 'brown', 'cyan']
    
    # Use modulo to avoid index out of range
    return colors[label % len(colors)]

def save_combined_image(dataset_name, calib_data, boxes, scores, labels, dimensions, orientation_angle, depth, image_path, output_image_path):


    if dataset_name == 'kitti':
        default_calib_folder = default_kitti_calib_folder
        default_label_folder = default_kitti_label_folder
    elif dataset_name == 'waymo':
        default_calib_folder = default_waymo_calib_folder
        default_label_folder = default_waymo_label_folder
    
    # Open the original image
    image_gt = Image.open(image_path)
    draw_gt = ImageDraw.Draw(image_gt)

    image_pred = Image.open(image_path)
    draw_pred = ImageDraw.Draw(image_pred)

    P = calib_data[0]
    
        # Get label file path
    label_file = os.path.splitext(os.path.basename(image_path))[0] + '.txt'
    label_path = os.path.join(default_label_folder, label_file)

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
                print("bbox, dimension_3d, location_3d, rotation_y", bbox, dimension_3d, location_3d, rotation_y)
                draw_3d_box(draw_gt, corners_2d)

    # Draw prediction bounding boxes in green
    #for box, score, label in zip(boxes, scores, labels):
    #print("dimensions out of loop", dimensions)
    for box, score, label, dimension, orientation_angle, depth in zip(boxes, scores, labels, dimensions, orientation_angle, depth):
        if score > 0.5:  # Threshold can be adjusted
            #draw_pred.rectangle(box.tolist(), outline="blue")  # Draw 2D bounding box in blue
            #print("dimension in the loop", dimension)
            
            # Draw 3D box
            #corners_3d = create_3d_bbox(dimension, location, orientation)

            #corners_2d = project_to_image(corners_3d.T, P)
            #print("corners_2d", corners_2d)
            #draw_3d_box(draw_gt, corners_2d, color="blue")

            corners_3d = create_3d_bbox_2d(dimension, box, depth, orientation_angle, P)
            #print("bbox", box)
            #print("corners_3d", corners_3d)
            corners_2d = project_to_image(corners_3d.T, P)
            #print("corners_2d", corners_2d)
            #draw_3d_box(draw_pred, corners_2d, color="green")
            #print("label", label)
            box_color = get_color_for_label(label.item())  # Get color based on the label
            
            draw_3d_box(draw_pred, corners_2d, color=box_color)
    
    # Save the combined image
    #image.save(output_image_path)

    # Combine the two images (stack them vertically)
    combined_image = Image.new('RGB', (image_gt.width, image_gt.height * 2))
    combined_image.paste(image_gt, (0, 0))
    combined_image.paste(image_pred, (0, image_gt.height))

    # Save the combined image
    combined_image.save(output_image_path)


def main(mode='inference', dataset_name='waymo', load=None):
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
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=1)

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

        for batch_idx, batch_data in enumerate(data_loader):

            if num_images and processed_images >= num_images:
                break  # Stop if we have processed the desired number of images

            if batch_data is None:
                continue  # Skip this iteration if batch_data is None

            images, _, calib_data, image_paths = batch_data

            images = images.to(device)

            original_image_path = image_paths[0]
            
            for idx, image in enumerate(images):
                image_path = image_paths[idx]
                single_image_predictions = []
                single_image_ground_truths = []

                # Perform inference on single image
                detections = inference(model, image_path, dataset_name)
                boxes, scores, labels, dimensions, orientation_angle, depth = detections

                print(detections)
                original_image_path = image_paths[0]
                # Assuming the variable 'calib_data' contains the tuple mentioned
                #print("calib_data", calib_data)
                #PIL_image, _, calibration_matrix, image_path = calib_data

                save_combined_image(dataset_name, calib_data, boxes, scores, labels, dimensions, orientation_angle, depth, original_image_path, f"{default_output_image_path}/output_{batch_idx:03d}.png")


                processed_images += 1

        generate_gif(default_output_image_path, f"{default_output_image_path}.gif")
        print(f"Completed inference on {processed_images} images.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FCOS Training/Inference')
    parser.add_argument('--mode', type=str, default='inference', choices=['train', 'inference'], help='Mode to run the script in')
    parser.add_argument('--dataset', type=str, default='waymo', choices=['kitti', 'waymo'], help='Dataset to use')
    parser.add_argument('--load', default=default_load_checkpoint, type=str, help='Path to the pretrained model file')
    args = parser.parse_args()
    main(mode=args.mode, dataset_name=args.dataset, load=args.load)
