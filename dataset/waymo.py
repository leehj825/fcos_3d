from torch.utils.data import Dataset
from PIL import Image
import os


import csv
import os
from typing import Any, Callable, List, Optional, Tuple

from PIL import Image

from .vision import VisionDataset
import numpy as np

# Reference:
# https://github.com/pytorch/vision/blob/main/torchvision/datasets/kitti.py
class Waymo(VisionDataset):
    """`Waymo <http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark>`_ Dataset.

    It corresponds to the "left color images of object" dataset, for object detection.

    Args:
        root (string): Root directory where images are downloaded to.
            Expects the following folder structure if download=False:

            .. code::

                <root>
                    └── Kitti
                        └─ raw
                            ├── training
                            |   ├── image_0
                            |   └── label_0
                            └── testing
                                └── image_0
        train (bool, optional): Use ``train`` split if true, else ``test`` split.
            Defaults to ``train``.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample
            and its target as entry and returns a transformed version.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    data_url = "none"
    resources = [
        "data_object_image_2.zip",
        "data_object_label_2.zip",
    ]
    image_dir_name = "image_0"   # front camera
    labels_dir_name = "label_0"
    calib_dir_name = "calib"
    calib_data_key = "P0"

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        download: bool = False,
    ):
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )
        self.images = []
        self.targets = []
        self.calibrations = []
        self.root = root
        self.train = train
        self._location = "training" if self.train else "testing"

        #if download:
        #    self.download()
        if not self._check_exists():
            raise RuntimeError("Dataset not found. You may use download=True to download it.")

        image_dir = os.path.join(self.root, self._location, self.image_dir_name)
        calib_dir = os.path.join(self.root, self._location, self.calib_dir_name)
        if self.train:
            labels_dir = os.path.join(self.root, self._location, self.labels_dir_name)
        for img_file in os.listdir(image_dir):
            self.images.append(os.path.join(image_dir, img_file))
            self.calibrations.append(os.path.join(calib_dir, f"{img_file.split('.')[0]}.txt"))
            if self.train:
                self.targets.append(os.path.join(labels_dir, f"{img_file.split('.')[0]}.txt"))

 

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any, str]:
        """Get item at a given index.

        Args:
            index (int): Index
        Returns:
            tuple: (image, target, calib_data, image_path), where
            target is a list of dictionaries with various keys. If label or calibration file is missing, returns None.
        """
        image_path = self.images[index]  # Get the path of the image

        # Check if calibration file exists (if necessary)
        calib_file = self.calibrations[index]

        if self.train:
            # Check if label file exists
            label_file = self.targets[index]
            if not os.path.exists(label_file):
                print(f"Label file missing: {label_file}")
                return None, None, calib_data, image_path

        image = Image.open(image_path)
        calib_data = self._parse_calibration(index)

        if self.train:
            target = self._parse_target(index)

        if self.transforms:
            image, target = self.transforms(image, target)

        if not self.train:
            return image, None, calib_data, image_path
            
        return image, target, calib_data, image_path

    def _parse_calibration(self, index: int):
        with open(self.calibrations[index], 'r') as f:
            for line in f:
                # Skip empty lines
                if not line.strip():
                    continue

                key, value = line.split(':', 1)
                if key == self.calib_data_key:
                    # Found P0, process and return it
                    values = [float(x) for x in value.split()]
                    calib_data = np.array(values).reshape(3, 4)
                    return calib_data

        # If P0 is not found in the file, handle it appropriately
        # For example, you can return None or raise an error
        return None

    def _parse_target(self, index: int) -> List:
        target = []
        with open(self.targets[index]) as inp:
            content = csv.reader(inp, delimiter=" ")
            for line in content:
                target.append(
                    {
                        "type": line[0],
                        "truncated": float(line[1]),
                        "occluded": int(line[2]),
                        "alpha": float(line[3]),
                        "bbox": [float(x) for x in line[4:8]],
                        "dimensions": [float(x) for x in line[8:11]],
                        "location": [float(x) for x in line[11:14]],
                        "rotation_y": float(line[14]),
                    }
                )
        return target

    def __len__(self) -> int:
        return len(self.images)

    @property
    def _raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    def _check_exists(self) -> bool:
        """Check if the data directory exists."""
        folders = [self.image_dir_name]
        if self.train:
            folders.append(self.labels_dir_name)
        return all(os.path.isdir(os.path.join(self.root, self._location, fname)) for fname in folders)




#class KITTIDataset(Dataset):
#    CLASS_MAPPING = {
#        "Pedestrian": 0,
#        "Cyclist": 1,
#        "Car": 2,
#        "Van": 3,
#        "Truck": 4,
#        "Person_sitting": 5,
#        "Tram": 6,
#        "Misc": 7,
#        "DontCare": 8,
#        # add other classes if needed
#    }
#    def __init__(self, root_dir, transform=None):
#        self.root_dir = root_dir
#        self.transform = transform
#        self.image_files = sorted(os.listdir(os.path.join(root_dir, "image_2")))
#        self.label_files = sorted(os.listdir(os.path.join(root_dir, "label_2")))
#
#
#    def __len__(self):
#        return len(self.image_files)
#
#    def __getitem__(self, idx):
#        img_name = os.path.join(self.root_dir, "image_2", self.image_files[idx])
#        image = Image.open(img_name)
#
#        label_name = os.path.join(self.root_dir, "label_2", self.label_files[idx])
#        with open(label_name, 'r') as f:
#            lines = f.readlines()
#
#        # Parse KITTI annotations
#        bboxes = []
#        classes = []
#        for line in lines:
#            data = line.split()
#            # Here, you'll have to adjust based on what classes you're interested in and
#            # how you want to represent them.
#            classes.append(data[0])
#            bboxes.append(list(map(float, data[4:8])))  # 2D bounding box
#
#        #print(classes)
#        #print(bboxes)
#        int_classes = [self.CLASS_MAPPING[cls] for cls in classes]
#        if self.transform:
#            image, bboxes, classes = self.transform(image, bboxes, int_classes)
#
#        return image, bboxes, classes

