# gv-benchmark for Detection

## CityPerson

### 准备

- 模型准备

  下载模型，保存到`./checkpoints/`

- 数据准备

  将cityperson数据集链接到 `./data/cityperson/`
  ```
  $ ln -s /xx/your_cityperson/leftImg8bit_trainvaltest ./data/cityperson/leftImg8bit_trainvaltest
  ```
  然后，将以下三个标注文件下载到`./data/cityperson/`
  ```
  http://10.5.8.69:39926/lists/train_10p_x10.json
  http://10.5.8.69:39926/lists/val_gt.json
  http://10.5.8.69:39926/lists/val_gt_for_mmdetction.json
  ```

### 运行

```
$ ./tools/mim_slurm_train.sh <your_partition> mmdet ./configs/det/linear_probe/faster_rcnn/r50_fpn_imagenet_pretrain_10p_cityperson.py output/CityPerson/xx/
```
