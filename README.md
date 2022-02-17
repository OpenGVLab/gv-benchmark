# Introduction

* We build GV-B(General Vision Benchmark) on Classification, Detection, Segmentation and Depth Estimation including 26 datasets for model evaluation.
* It is recommended to evaluate with low-data regime, using only 10% training data.
* The parameters of model backbone will be frozen during training, as known as 'linear probe'.
* Face Detection and Depth Estimation is not provided for now, you may evaluate via official repo if needed.
* Specifically, we use central_model.py in our repo to represent the implementation of Up-G models.

## Task Supported

* [x] Object Classification
* [x] Object Detection (VOC Detection)
* [x] Pedestrian Detection (CityPersons Detection)
* [x] Semantic Segmentation (VOC Segmentation)
* [ ] Face Detection (WiderFace Detection)
* [ ] Depth Estimation (Kitti/NYU-v2 Depth Estimation)

# Installation

## Requirements

- Python 3.6+
- PyTorch 1.8+
- openmim==0.1.5
- [MMCV>=1.3.16](https://github.com/open-mmlab/mmcv)


## Install Dependencies

a. Create a conda virtual environment and activate it.

```shell
conda create -n open-mmlab python=3.8 -y
conda activate open-mmlab
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.：

```shell
conda install pytorch torchvision -c pytorch
```

```{note}
Make sure that your compilation CUDA version and runtime CUDA version match.
You can check the supported CUDA version for precompiled packages on the
[PyTorch website](https://pytorch.org/).
```

c. Install openmm package via pip (mmcls, mmdet, mmseg)：

```shell
pip install mmcls
pip install mmdet
pip install mmsegmentation
```


# Usage

This section provide basic tutorials about the usage of GV-B.


## Prepare datasets

For each evaluation task, you can follow the official repo tutorial for data preparation.

[mmclassification](https://github.com/open-mmlab/mmclassification/blob/master/docs/zh_CN/getting_started.md)

[mmdetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/zh_cn/1_exist_data_model.md)

[mmsegmentation](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/zh_cn/dataset_prepare.md)


## Model evaluation

We use [MIM](https://github.com/open-mmlab/mim) to submit evaluation in GV-B.

a.If you run on a cluster managed with slurm, you can use the script mim_slurm_train.sh. (This script also supports single machine training.)

```shell
sh tools/mim_slurm_train.sh $PARTITION $TASK $CONFIG $WORK_DIR

## mmcls as an example
sh tools/mim_slurm_train.sh GVT mmcls configs/cls/linear_probe/mnb4_Up-E-C_pretrain_flowers_10p.py /path/to/your/project
```


b.If you run on w/o slurm. (More details can be found in docs of openmim)

```bash
PYTHONPATH='.':$PYTHONPATH mim train $TASK $CONFIG $WORK_DIR
```

- PARTITION: The partition you are using
- WORK_DIR: The directory to save logs and checkpoints
- CONFIG: Config files corresponding to tasks.


## Detailed Tutorials

Currently, we provide tutorials for users.

- [add new modules](https://mmclassification.readthedocs.io/en/latest/tutorials/new_modules.html)
- [add new dataset](https://mmclassification.readthedocs.io/en/latest/tutorials/new_dataset.html)
- [learn about config](https://mmclassification.readthedocs.io/en/latest/tutorials/config.html)
- [design data pipeline](https://mmclassification.readthedocs.io/en/latest/tutorials/data_pipeline.html)
- [customize schedule](https://mmclassification.readthedocs.io/en/latest/tutorials/schedule.html)
- [customize runtime settings](https://mmclassification.readthedocs.io/en/latest/tutorials/runtime.html)


## Benchmark(with Hyperparameter searching)

|          |           | CLS     |          |      |      |         |      |      |      |         |          |      |         |          |             |         |        |       |      |          |             | DET      |                |             | SEG     | DEP   |       |
| -------- | --------- | ------- | -------- | ---- | ---- | ------- | ---- | ---- | ---- | ------- | -------- | ---- | ------- | -------- | ----------- | ------- | ------ | ----- | ---- | -------- | ----------- | -------- | -------------- | ----------- | ------- | ----- | ----- |
| 10% data |           | Cifar10 | Cifar100 | Food | Pets | Flowers | Sun  | Cars | Dtd  | Caltech | Aircraft | Svhn | Eurosat | Resisc45 | Retinopathy | Fer2013 | Ucf101 | Gtsrb | Pcam | Imagenet | Kinetics700 | VOC07+12 | WIDER  FACE    | CityPersons | VOC2012 | KITTI | NYUv2 |
| Up-A     | R50       | 92.4    | 73.5     | 75.8 | 85.7 | 94.6    | 57.9 | 52.7 | 65.0 | 88.5    | 28.7     | 61.4 | 93.8    | 82.9     | 73.8        | 55.0    | 71.1   | 75.1  | 82.9 | 71.9     | 35.2        | 76.3     | 90.3/88.3/70.7 | 24.6/59.0   | 62.54   | 3.181 | 0.456 |
|          | MN-B4     | 96.1    | 82.9     | 84.3 | 89.8 | 98.3    | 66.0 | 61.4 | 66.8 | 92.8    | 32.5     | 60.4 | 92.7    | 85.8     | 75.6        | 56.5    | 76.9   | 74.4  | 84.3 | 77.2     | 39.4        | 74.9     | 89.3/87.6/71.4 | 26.5/61.8   | 65.71   | 3.565 | 0.482 |
|          | MN-B15    | 98.2    | 87.8     | 93.9 | 92.8 | 99.6    | 72.3 | 59.4 | 70.0 | 93.8    | 64.8     | 58.6 | 95.3    | 91.9     | 77.9        | 62.8    | 85.4   | 76.2  | 87.8 | 86.0     | 52.9        | 78.4     | 93.6/91.8/77.2 | 17.7/49.5   | 60.68   | 2.423 | 0.383 |
| Up-E     | C-R50     | 91.9    | 71.2     | 80.7 | 88.8 | 94.0    | 57.4 | 67.9 | 62.7 | 85.5    | 73.9     | 57.6 | 93.7    | 83.6     | 75.4        | 54.1    | 69.6   | 73.9  | 85.7 | 72.5     | 34.6        | 72.2     | 89.7/87.6/68.1 | 22.4/58.3   | 57.66   | 3.214 | 0.501 |
|          | D-R50     | 86.4    | 57.3     | 53.9 | 31.4 | 44.0    | 39.8 | 8.6  | 44.6 | 72.5    | 15.8     | 64.2 | 89.1    | 72.8     | 73.6        | 46.6    | 57.4   | 67.5  | 81.7 | 45.0     | 25.2        | 87.7     | 93.8/92.0/75.5 | 15.8/41.5   | 62.3    | 3.09  | 0.45  |
|          | S-R50     | 78.3    | 46.6     | 45.1 | 24.2 | 33.9    | 38.0 | 5.0  | 41.4 | 50.2    | 8.5      | 51.5 | 89.9    | 76.4     | 74.0        | 44.8    | 42.0   | 64.0  | 80.8 | 34.9     | 19.7        | 75.0     | 87.4/85.7/66.4 | 19.6/53.3   | 71.9    | 3.12  | 0.45  |
|          | C-MN-B4   | 96.7    | 83.2     | 89.2 | 91.9 | 98.2    | 66.7 | 67.7 | 66.3 | 91.9    | 77.2     | 57.8 | 94.4    | 88.0     | 77.0        | 56.6    | 78.5   | 77.3  | 85.6 | 80.5     | 44.2        | 73.7     | 89.6/88.0/71.1 | 30.3/65.0   | 65.8    | 3.54  | 0.46  |
|          | D-MN-B4   | 91.5    | 67.0     | 61.4 | 44.4 | 57.2    | 41.8 | 12.1 | 41.2 | 80.6    | 25.1     | 68.0 | 90.7    | 74.6     | 74.3        | 50.3    | 61.7   | 74.2  | 81.9 | 57.0     | 29.3        | 89.3     | 94.6/92.6/76.5 | 14.0/43.8   | 73.1    | 3.05  | 0.40  |
|          | S-MN-B4   | 83.5    | 57.2     | 68.3 | 70.8 | 85.8    | 52.9 | 25.9 | 52.8 | 81.6    | 17.7     | 56.1 | 91.3    | 83.6     | 74.5        | 49.0    | 55.2   | 68.0  | 84.3 | 61.0     | 27.4        | 78.7     | 89.5/87.9/71.4 | 19.4/53.0   | 79.6    | 3.06  | 0.41  |
|          | C-MN-B-15 | 98.7    | 90.1     | 94.7 | 95.1 | 99.7    | 75.7 | 74.9 | 73.6 | 94.4    | 91.8     | 66.7 | 96.2    | 92.8     | 77.6        | 62.3    | 87.7   | 83.3  | 87.5 | 87.2     | 54.7        | 80.4     | 93.2/91.4/75.7 | 29.5/59.9   | 70.6    | 2.63  | 0.37  |
|          | D-MN-B-15 | 92.2    | 67.9     | 69.0 | 33.9 | 59.5    | 45.4 | 13.8 | 46.3 | 82.0    | 26.6     | 65.4 | 90.1    | 79.1     | 76.0        | 53.2    | 63.7   | 74.4  | 83.3 | 62.2     | 33.7        | 89.4     | 95.8/94.4/80.1 | 10.5/42.4   | 77.2    | 2.72  | 0.37  |
| Up-G     | R50       | 92.9    | 73.7     | 81.1 | 88.9 | 94.0    | 58.6 | 68.6 | 63.0 | 86.1    | 74.0     | 57.9 | 94.4    | 84.0     | 75.7        | 54.3    | 70.8   | 74.3  | 85.9 | 72.6     | 34.8        | 87.7     | 93.9/92.2/77.0 | 14.7/46.0   | 66.19   | 2.835 | 0.39  |
|          | MN-B4     | 96.7    | 83.9     | 89.2 | 92.1 | 98.2    | 66.7 | 67.7 | 66.5 | 91.9    | 77.2     | 57.8 | 94.4    | 88.0     | 77.0        | 57.1    | 79     | 77.7  | 86   | 80.5     | 44.2        | 89.1     | 94.9/92.8/76.5 | 12.0/50.5   | 71.4    | 2.94  | 0.40  |
|          | MN-B15    | 98.7    | 90.4     | 94.5 | 95.4 | 99.7    | 74.4 | 75.4 | 74.2 | 94.5    | 91.8     | 66.7 | 96.3    | 92.7     | 77.9        | 63.1    | 88     | 83.6  | 88   | 87.1     | 54.7        | 89.8     | 95.9/94.2/79.6 | 10.5/41.3   | 77.3    | 2.71  | 0.37  |
