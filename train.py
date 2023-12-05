import torch
import torchvision
import os
import argparse
from PIL import Image
from tqdm.auto import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import dataset.kitti as kitti
import model.fcos3d as fcos3d
from PIL import Image, ImageDraw
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchvision.models.resnet import ResNet101_Weights

import numpy as np


# Set environment variable for MPS (if using Apple Silicon)
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# Define class mapping for the KITTI dataset
CLASS_MAPPING = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}

default_data_path="data/kitti_200/"

default_learning_rate = 0.001

default_image_path ='/Users/hyejunlee/NN_Scratch/data/kitti/training/image_2/000025.png'
#default_load_checkpoint = '/Users/hyejunlee/NN_Scratch/save_state_2d_220.bin'
default_load_checkpoint = None

default_output_image_path = 'output_save_state_2d_220_25.png' 


default_label_folder ='/Users/hyejunlee/NN_Scratch/data/kitti/training/label_2/'
default_calib_folder ='/Users/hyejunlee/NN_Scratch/data/kitti/training/calib/'

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
#device = "cpu"
print(device)

def custom_collate(batch):
    resize_transform = transforms.Compose([
        transforms.Resize((375, 1242)),
        transforms.ToTensor()
    ])
    images, targets, calib_data, image_path = zip(*batch)
    
    images = [resize_transform(img) for img in images]
    images = torch.stack(images, 0).to(device)

    # Calibration data is already in the required format, so we just pass it as is
    # If any further processing is needed, add it here

    return images, targets, calib_data, image_path

def preprocess_image(img_path):
    # Preprocess the image for inference
    #image = Image.open(img_path).convert("RGB")
    image = Image.open(img_path)
    resize_transform = transforms.Compose([
        transforms.Resize((375, 1242)),
        transforms.ToTensor()
    ])
    image = resize_transform(image).unsqueeze(0).to(device)
    return image

def inference_old(model, image_path):
    # Perform inference on a given image
    model.eval()
    with torch.no_grad():
        image = preprocess_image(image_path)
        prediction = model(image)
        # Post-process the output if necessary
    return prediction

def inference(model, image_path):
    # Perform inference on a given image
    model.eval()
    with torch.no_grad():
        image = preprocess_image(image_path)
        predictions = model(image)
        # Example post-processing (adjust as per your model's output format)
        # Assuming predictions is a list of dictionaries with 'boxes', 'scores', 'labels', 'dimensions', 'locations', 'rotations' keys
        if predictions:
            pred = predictions[0]  # Assuming batch size of 1
            boxes = pred['boxes']
            scores = pred['scores']
            labels = pred['labels']
            dimensions = pred['dimensions_3d']
            locations = pred['locations_3d']
            orientation = pred['orientation']
        else:
            boxes, scores, labels, dimensions, locations, orientation = [], [], [], [], [], []
    return boxes, scores, labels, dimensions, locations, orientation


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
        draw.line([tuple(corners_2d[i]), tuple(corners_2d[j])], fill=color, width=2)


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

# Rest of the functions (project_to_image, draw_3d_box) remain the same as in the previous example

def save_combined_image(boxes, scores, labels, dimensions, locations, orientation, image_path, output_image_path):
    # Open the original image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Get label file path
    calib_file = os.path.splitext(os.path.basename(image_path))[0] + '.txt'
    calib_path = os.path.join(default_calib_folder, calib_file)


    # Read calibration data
    calib_data = {}
    with open(calib_path, 'r') as calib_file:
        for line in calib_file:
            if ':' in line:  # Ensure line contains a colon
                key, value = line.split(':', 1)
                calib_data[key] = np.fromstring(value, sep=' ')

    # Extract P2 calibration matrix
    P = calib_data.get('P2').reshape(3, 4) if 'P2' in calib_data else None
    if P is None:
        raise ValueError("P2 calibration matrix not found in calibration data.")

    # Draw ground truth bounding boxes in yellow and 3D bounding boxes
    with open(default_label_folder + os.path.basename(image_path).replace('.png', '.txt'), 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            bbox = [float(parts[i]) for i in range(4, 8)]  # 2D bounding box
            dimension_3d = [float(parts[i]) for i in range(8, 11)]  # 3D dimensions
            location_3d = [float(parts[i]) for i in range(11, 14)]  # 3D location
            rotation_y = float(parts[14])  # Orientation

            corners_3d = create_3d_bbox(dimension_3d, location_3d, rotation_y)
            corners_2d = project_to_image(corners_3d.T, P)

            #draw.rectangle(bbox, outline="yellow")
            draw_3d_box(draw, corners_2d)

    # Draw prediction bounding boxes in green
    #for box, score, label in zip(boxes, scores, labels):
    #print("dimensions out of loop", dimensions)
    for box, score, label, dimension, location, orientation in zip(boxes, scores, labels, dimensions, locations, orientation):
        if score > 0.5:  # Threshold can be adjusted
            #draw.rectangle(box.tolist(), outline="green")
            #print("dimension in the loop", dimension)
            
            # Draw 3D box
            corners_3d = create_3d_bbox(dimension, location, orientation)
            corners_2d = project_to_image(corners_3d.T, P)
            draw_3d_box(draw, corners_2d, color="green")

    # Save the combined image
    image.save(output_image_path)


def main(mode='train', image_path=None, load=None):
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

    # Load the KITTI dataset
    dataset = kitti.Kitti(root=default_data_path, train=True)
    
    # Define a data loader
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)

    num_epochs = 500
    start_epoch = 0

    # Load model checkpoint if provided
    if load:
        checkpoint = torch.load(load)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if mode == 'train':
        # Training loop
        for epoch in range(start_epoch, num_epochs):
            for batch_idx, (images, targets, calib_data, image_paths) in enumerate(data_loader):

                # Extract the original image size (height, width) from the images tensor
                _, _, orig_height, orig_width = images.size()

                # Prepare targets for each image
                target_list = []
                for image_targets in targets:
                    boxes = []  # 2D bounding boxes
                    labels = []    # Object labels
                    dimensions_3d = []  # 3D bounding box dimensions (height, width, length)
                    locations_3d = []  # 3D bounding box center location (x, y, z)
                    orientations_y = []  # Rotation around the Y-axis

                    for target in image_targets:
                        if target["type"] in CLASS_MAPPING:
                            # Parsing the 2D bounding box and object label
                            boxes.append(target["bbox"])
                            labels.append(CLASS_MAPPING[target["type"]])

                            # Parsing 3D bounding box information
                            dimensions_3d.append(target["dimensions"])  # [height, width, length]
                            locations_3d.append(target["location"])     # [x, y, z]
                            orientations_y.append(target["rotation_y"]) # rotation y

                    target_dict = {
                        'boxes': torch.tensor(boxes, dtype=torch.float32).to(device).reshape(-1, 4),
                        'labels': torch.tensor(labels, dtype=torch.int64).to(device),
                        'dimensions_3d': torch.tensor(dimensions_3d, dtype=torch.float32).to(device),
                        'locations_3d': torch.tensor(locations_3d, dtype=torch.float32).to(device),
                        'orientations_y': torch.tensor(orientations_y, dtype=torch.float32).to(device)
                    }
                    target_list.append(target_dict)
                #print("target_list: ", target_list)

                # Forward pass
                loss_dict = model(images, target_list)

                # Print each loss component for the current batch
                loss_info = f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(data_loader)}, "

                for loss_name, loss_value in loss_dict.items():
                    loss_info += f"{loss_name}: {'{:.3f}'.format(loss_value.item())}, "

                print(loss_info.strip(", "))

                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()

                # Backward pass
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

            print(f"Epoch {epoch+1} of {num_epochs}, Loss: {loss_value}")

            # Save checkpoint
            if (epoch+1) % 1 == 0:
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, f"./save_state_2d_{epoch+1}.bin")

    elif mode == 'inference':
        device = "cpu"
        model = model.to(device)
        
        # Inference mode
        if image_path:
            detections = inference(model, image_path)
            boxes, scores, labels, dimensions, locations, rotations = detections

            #print(detections)
            save_combined_image(boxes, scores, labels, dimensions, locations, rotations, image_path, default_output_image_path)


            #print(detections)
        else:
            print("Please provide an image path for inference.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FCOS Training/Inference')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'], help='Mode to run the script in')
    parser.add_argument('--image_path', default=default_image_path, type=str, help='Path to the image for inference mode')
    parser.add_argument('--load', default=default_load_checkpoint, type=str, help='Path to the pretrained model file')
    args = parser.parse_args()
    main(mode=args.mode, image_path=args.image_path, load=args.load)
