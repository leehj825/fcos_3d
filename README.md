# FCOS+3d

description

## Installation 

### CUDA and Pytorch
```
conda install -c conda-forge cudatoolkit=11.7.0
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
```
```
### In SJSU HPC, enable GPU node and confirm CUDA is enabled
```
srun -p gpu --gres=gpu -n 1 -N 1 -c 4 --pty /bin/bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Models and Code Bases

- **Model**:
   - FCOS + 3D

- **Dataset**
    - Kitti
    - Waymo

## Dataset Preparation
Kitti Dataset for 3D Object Detection: https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d

### Kitti 2017 Dataset without data processing
Download **kitti_data.zip** from below link and unzip.
- https://drive.google.com/file/d/1r_kvJ2zTgeu6X5QUtSaYa9PIpauz0Hrh/view?usp=sharing

To simulate federated learnings, datasets are separated into 4 groups
```
│   ├── kitti_1
│   ├── kitti_2
│   ├── kitti_3
│   ├── kitti_4
```
For creating new Kitti dataset or converting Waymo dataset, please refer to Appendix below.

## Training
```
```

```
Training output
```

```
### Loss output from Training


## Reference


## Appendix

### Create Kitti Format Annotation
Refer: https://mmdetection3d.readthedocs.io/en/latest/user_guides/dataset_prepare.html

Download Kitti dataset to below folder structure
```
│   ├── kitti
│   │   ├── ImageSets
│   │   ├── testing
│   │   │   ├── calib
│   │   │   ├── image_2
│   │   │   ├── velodyne
│   │   ├── training
│   │   │   ├── calib
│   │   │   ├── image_2
│   │   │   ├── label_2
│   │   │   ├── velodyne
```

#### Script to create kitti pickle annotation files
```
python tools/create_data.py kitti --root-path ./data/kitti --out-dir ./data/kitti --extra-tag kitti
```
3 Classes are defined in annotation files.
```
+------------+--------+
| category   | number |
+------------+--------+
| Pedestrian | 2207   |
| Cyclist    | 734    |
| Car        | 14352  |
+------------+--------+
```
### Convert Waymo tfrecord to kitti format
Download tfrecords (1.4.1) from here https://waymo.com/open/download/

#### Waymo format structure
```
│   ├── waymo
│   │   ├── waymo_format (each subfolder containing tfrecord files)
│   │   │   ├── testing
│   │   │   ├── testing_3d_camera_only_detection
│   │   │   ├── training
│   │   │   ├── validation
│   │   ├── kitti_format (output files from script)
```
#### Convert to kitti format
```
> python tools/create_data.py waymo --root-path data/waymo --out-dir data/waymo --extra-tag waymo

:
:

tfrecord_pathnames(0) data/waymo/waymo_format/training/segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord
tfrecord_pathnames(1) data/waymo/waymo_format/training/segment-10023947602400723454_1120_000_1140_000_with_camera_labels.tfrecord
2023-10-12 06:59:41.883071: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
2023-10-12 06:59:41.885273: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
tfrecord_pathnames(3) data/waymo/waymo_format/training/segment-10061305430875486848_1080_000_1100_000_with_camera_labels.tfrecord
2023-10-12 06:59:41.910530: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
tfrecord_pathnames(4) data/waymo/waymo_format/training/segment-10072140764565668044_4060_000_4080_000_with_camera_labels.tfrecord
tfrecord_pathnames(5) data/waymo/waymo_format/training/segment-10072231702153043603_5725_000_5745_000_with_camera_labels.tfrecord
[>>                                                ] 1/21, 0.0 task/s, elapsed: 562s, ETA: 11240stfrecord_pathnames(6) data/waymo/waymo_format/training/segment-10075870402459732738_1060_000_1080_000_with_camera_labels.tfrecord
[>>>>>>>>>                                         ] 4/21, 0.0 task/s, elapsed: 622s, ETA:  2644stfrecord_pathnames(7) data/waymo/waymo_format/training/segment-10082223140073588526_6140_000_6160_000_with_camera_labels.tfrecord
tfrecord_pathnames(8) data/waymo/waymo_format/training/segment-10094743350625019937_3420_000_3440_000_with_camera_labels.tfrecord
[>>>>>>>>>>>>>>                                    ] 6/21, 0.0 task/s, elapsed: 1107s, ETA:  2767stfrecord_pathnames(9) data/waymo/waymo_format/training/segment-10096619443888687526_2820_000_2840_000_with_camera_labels.tfrecord
[>>>>>>>>>>>>>>>>                                  ] 7/21, 0.0 task/s, elapsed: 1124s, ETA:  2248stfrecord_pathnames(10) data/waymo/waymo_format/training/segment-10107710434105775874_760_000_780_000_with_camera_labels.tfrecord
[>>>>>>>>>>>>>>>>>>>                               ] 8/21, 0.0 task/s, elapsed: 1184s, ETA:  1924stfrecord_pathnames(11) data/waymo/waymo_format/training/segment-10153695247769592104_787_000_807_000_with_camera_labels.tfrecord
[>>>>>>>>>>>>>>>>>>>>>                             ] 9/21, 0.0 task/s, elapsed: 1636s, ETA:  2182stfrecord_pathnames(12) data/waymo/waymo_format/training/segment-1022527355599519580_4866_960_4886_960_with_camera_labels.tfrecord
[>>>>>>>>>>>>>>>>>>>>>>>                           ] 10/21, 0.0 task/s, elapsed: 1669s, ETA:  1836stfrecord_pathnames(13) data/waymo/waymo_format/training/segment-10231929575853664160_1160_000_1180_000_with_camera_labels.tfrecord
[>>>>>>>>>>>>>>>>>>>>>>>>>>                        ] 11/21, 0.0 task/s, elapsed: 1677s, ETA:  1524stfrecord_pathnames(14) data/waymo/waymo_format/training/segment-10235335145367115211_5420_000_5440_000_with_camera_labels.tfrecord
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>                      ] 12/21, 0.0 task/s, elapsed: 1725s, ETA:  1294stfrecord_pathnames(15) data/waymo/waymo_format/training/segment-10241508783381919015_2889_360_2909_360_with_camera_labels.tfrecord
```
#### Waymo dataset Output in Kitti Format

```
│   ├── kitti_format
│   │   ├── ImageSets
│   │   ├── testing_3d_camera_only_detection
│   │   │   ├── image_0
│   │   │   ├── image_1
│   │   │   ├── image_2
│   │   │   ├── image_3
│   │   │   ├── [others that are not used for this project]
│   │   ├── training
│   │   │   ├── image_0
│   │   │   ├── image_1
│   │   │   ├── image_2
│   │   │   ├── image_3
│   │   │   ├── label_0
│   │   │   ├── label_1
│   │   │   ├── label_2
│   │   │   ├── label_3
│   │   │   ├── [others that are not used for this project]
│   │   ├── waymo_infos_test.pkl
│   │   ├── waymo_infos_train.pkl
│   │   ├── waymo_infos_trainval.pkl
│   │   ├── waymo_infos_val.pkl
```
Object label example in Waymo dataset
```
+------------+--------+
| category   | number |
+------------+--------+
| Car        | 33375  |
| Pedestrian | 8556   |
| Cyclist    | 294    |
+------------+--------+
```


## Troubleshoot

