### 准备数据集

#### 分类任务数据集准备

*对于retinopathy,  resisc45, eurosat, patchcamelyon四个数据集，我们参考了VTAB的数据获取与处理方式，为了与list配合使用请按如下流程准备对应的数据集

```
1.  安装vtab 参考https://github.com/google-research/task_adaptation
2.  按照Dataset preparation部分下载对应的数据集，属于tensorflow格式的数据，以tfrecord形式存在。
3.  用mmcls/tools/data/save_dataset_image.py调用vtab里面的api读取所有的图片并存储为png格式
```

*对于其他的16个数据集，可以自行去数据集官网下载并整理为list对应的目录结构，我们提供了10%数据对应的list
```
http://10.5.8.69:39926/lists/classification.zip
```

*建议将全部数据集放在任意同一个根目录下，对路径位置无要求

#### 检测数据集准备

- 模型准备
  下载模型，保存到`./checkpoints/`

- Citypersons数据准备
  将cityperson数据集链接到 `./data/cityperson/`
  ```
  $ ln -s /xx/your_cityperson/leftImg8bit_trainvaltest ./data/cityperson/leftImg8bit_trainvaltest
  ```
  然后，将以下三个标注文件下载到`./data/cityperson/`
  ```
  [train_10p_x10.json](http://10.5.8.69:39926/lists/train_10p_x10.json)
  [val_gt.json](http://10.5.8.69:39926/lists/val_gt.json)
  [val_gt_for_mmdetction.json](http://10.5.8.69:39926/lists/val_gt_for_mmdetction.json)
  ```
- Widerface数据集准备
  下载widerface数据集后，参考mmdet关于widerface部分的内容准备数据集对应的list，我们提供了对应的10%数据list
  ```
  http://10.5.8.69:39926/lists/widerface/10p.txt
  ```

对于其他测试任务(VOC检测/分割)，请按openmm对应的任务库文档进行数据集的准备，对应的10%数据list见
```
http://10.5.8.69:39926/lists/vocdet/
http://10.5.8.69:39926/lists/vocseg/
```

[mmclassification](https://github.com/open-mmlab/mmclassification/blob/master/docs/zh_CN/getting_started.md)

[mmdetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/zh_cn/get_started.md)

[mmsegmentation](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/zh_cn/get_started.md)
