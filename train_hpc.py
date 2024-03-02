import torch
import torchvision
import os
import argparse
from PIL import Image
from tqdm.auto import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import dataset.kitti as kitti
import dataset.waymo as waymo
import model.fcos3d as fcos3d
from PIL import Image, ImageDraw
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchvision.models.resnet import ResNet101_Weights

import numpy as np


# Set environment variable for MPS (if using Apple Silicon)
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# Define class mapping for the KITTI dataset
CLASS_MAPPING = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}

# Default paths and parameters for KITTI dataset
#default_kitti_data_path = "data/kitti_200/"
default_kitti_data_path = "/scratch/cmpe249-fa23/kitti_od/"
default_kitti_image_path = 'data/kitti_200/training/image_2/000025.png'
default_kitti_label_folder = 'data/kitti_200/training/label_2/'
default_kitti_calib_folder = 'data/kitti_200/training/calib/'

# Default paths and parameters for Waymo dataset
default_waymo_data_path = "/scratch/cmpe249-fa23/waymo_data/waymo_0000/kitti_format/"  # Update this path as per your Waymo dataset location
default_waymo_image_path = 'data/waymo_single/training/image_0/0000001.jpg'  # Update with a Waymo image path
default_waymo_label_folder = 'data/waymo_single/training/label_0/'
default_waymo_calib_folder = 'data/waymo_single/training/calib/'
# Add more Waymo specific paths and parameters if needed

default_learning_rate = 0.0001
#default_load_checkpoint = '/home/001891254/fcos_3d/save_state_waymo_depth_hpc_30.bin'
default_load_checkpoint = None
default_output_image_path = 'output_save_state_3.png'

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
#device = "cpu"
print(device)

def dump_trainable_params(model):
    total_params = 0
    print("Trainable Parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_shape = param.size()
            num_params = param.numel()
            total_params += num_params
            print(f"{name}: shape = {param_shape}, total params = {num_params}")
    print(f"Total trainable parameters: {total_params}")



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

def project_to_image(pts_3d, P):
    """ Project 3D points to 2D by applying the calibration matrix P """
    pts_3d_hom = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)))
    pts_2d_hom = np.dot(pts_3d_hom, np.transpose(P))  # Project
    pts_2d = pts_2d_hom[:, :2] / pts_2d_hom[:, 2:3]  # Normalize
    return pts_2d


def main(mode='train', dataset_name='waymo', image_path=None, load=None):
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

    num_epochs = 500
    start_epoch = 0

    # Load model checkpoint if provided
    if load:
        checkpoint = torch.load(load)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if mode == 'train':


        # Example usage:
        # Assuming 'model' is your FCOS model
        dump_trainable_params(model)

        # Training loop
        for epoch in range(start_epoch, num_epochs):
            
            total_loss = 0.0  # Initialize total loss for the epoch
            num_batches = 0  # Count the number of batches processed

            #for batch_idx, (images, targets, calib_data, image_paths) in enumerate(data_loader):
            for batch_idx, batch_data in enumerate(data_loader):
                if batch_data is None:
                    print("missing files")
                    continue
                #print("batch_data", batch_data)
                images, targets, calib_data, image_paths = batch_data
                    
                # Extract the original image size (height, width) from the images tensor
                _, _, orig_height, orig_width = images.size()
                #print("image_paths[0]", image_paths[0])

                # Prepare targets for each image
                target_list = []
                for image_targets in targets:
                    boxes = []  # 2D bounding boxes
                    labels = []    # Object labels
                    dimensions_3d = []  # 3D bounding box dimensions (height, width, length)
                    orientations_y = []  # Rotation around the Y-axis
                    depth = []  # 3D bounding box center location (x, y, z)

                    for target in image_targets:
                        if target["type"] in CLASS_MAPPING:
                            # Check if the bounding box has non-zero dimensions
                            bbox = target["bbox"]
                            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:  # Ensures width and height are positive
                                # Parsing the 2D bounding box and object label
                                boxes.append(bbox)
                                labels.append(CLASS_MAPPING[target["type"]])

                                # Parsing 3D bounding box information
                                dimensions_3d.append(target["dimensions"])  # [height, width, length]
                                orientations_y.append(target["rotation_y"]) # rotation y
                                depth.append(target["location"][2])

                    # Only proceed if there are valid targets
                    if boxes:
                        target_dict = {
                            'boxes': torch.tensor(boxes, dtype=torch.float32).to(device).reshape(-1, 4),
                            'labels': torch.tensor(labels, dtype=torch.int64).to(device),
                            'dimensions_3d': torch.tensor(dimensions_3d, dtype=torch.float32).to(device),
                            'orientations_y': torch.tensor(orientations_y, dtype=torch.float32).to(device),
                            'depth': torch.tensor(depth, dtype=torch.float32).to(device),
                        }
                        target_list.append(target_dict)

                #print("target_list: ", target_list)
                
                # Skip the rest of the loop if target_list is empty
                if not target_list:
                    continue

                # Forward pass
                loss_dict = model(images, target_list)

                # Print each loss component for the current batch
                loss_info = f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(data_loader)}, "
                
                # Print every 5 batches
                if (batch_idx + 1) % 10 == 0:
                    for loss_name, loss_value in loss_dict.items():
                        loss_info += f"{loss_name}: {'{:.3f}'.format(loss_value.item())}, "

                    print(loss_info.strip(", "))

                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()
                
                total_loss += loss_value  # Accumulate the loss from each batch
                num_batches += 1  # Increment the batch count

                # Backward pass
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

            # Update the learning rate scheduler
            scheduler.step(loss_value)
            # Optionally, log the current learning rate
            current_lr = scheduler.optimizer.param_groups[0]['lr']
            #print(f"Epoch {epoch+1}: Current learning rate: {current_lr}")

            #print(f"Epoch {epoch+1} of {num_epochs}, Loss: {loss_value}")
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            print(f"Epoch {epoch+1} of {num_epochs}, Learning Rate: {current_lr}, Avg Loss: {avg_loss:.4f}")


            # Save checkpoint
            if (epoch+1) % 5 == 0:
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, f"./save_state_{dataset_name}_new_depth_{epoch+1}.bin")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FCOS Training/Inference')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'], help='Mode to run the script in')
    parser.add_argument('--dataset', type=str, default='waymo', choices=['kitti', 'waymo'], help='Dataset to use')
    parser.add_argument('--image_path', default=default_waymo_image_path, type=str, help='Path to the image for inference mode')
    parser.add_argument('--load', default=default_load_checkpoint, type=str, help='Path to the pretrained model file')
    args = parser.parse_args()
    main(mode=args.mode, dataset_name=args.dataset, image_path=args.image_path, load=args.load)
