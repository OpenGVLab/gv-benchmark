import os
from collections import defaultdict

import cv2
import numpy as np
import task_adaptation.data_loader as data_loader


def get_data_params(dataset, is_full=True):
    if is_full:
        dataset_train_split_name = 'train'
        dataset_eval_split_name = 'test'
    else:
        dataset_train_split_name = 'train800val200'
        dataset_eval_split_name = 'val200'
    shuffle_buffer_size = 10000
    train_examples = None
    batch_size = 1
    batch_size_eval = 1
    data_dir = None
    input_range = [0.0, 1.0]
    return {
        'dataset': 'data.' + dataset,
        'dataset_train_split_name': dataset_train_split_name,
        'dataset_eval_split_name': dataset_eval_split_name,
        'shuffle_buffer_size': shuffle_buffer_size,
        'prefetch': 0,
        'train_examples': train_examples,
        'batch_size': batch_size,
        'batch_size_eval': batch_size_eval,
        'data_for_eval': True,
        'data_dir': data_dir,
        'input_range': [float(v) for v in input_range]
    }


datasets = ('patch_camelyon', 'resisc45', 'eurosat',
            'diabetic_retinopathy(config="btgraham-300")')

modes = ['train', 'eval']
is_full = False

os.system('make -p datainfo')
os.system('make -p datasetinfo/data')
os.system('make -p datainfo_1k')


def catimages(images):
    num = len(images)
    import math
    rows = cols = math.ceil(math.sqrt(num))
    for i in range(rows**2 - len(images)):
        images.append(np.ones(images[0].shape).astype(np.uint8) * 255)

    res = []
    for i in range(rows):
        res.append(np.concatenate(images[i * cols:(i + 1) * cols], axis=1))
    return np.concatenate(res, axis=0)[:, :, ::-1]


for dataset in datasets:
    label2images = defaultdict(list)
    label2cnt = dict()
    mode = 'train'
    data_params = get_data_params(dataset, is_full=is_full)

    for train_split, eval_split in [('train', 'test'), ('train800', 'val200'),
                                    ('train800', 'val')]:

        data_params['dataset_train_split_name'], data_params[
            'dataset_eval_split_name'] = train_split, eval_split
        data_params['dataset'] = data_loader.get_dataset_instance(data_params)
        for mode in ['train', 'eval']:
            input_fn_train = data_loader.build_data_pipeline(data_params,
                                                             mode=mode)
            data_fn = input_fn_train({'batch_size': 512})
            iter = 0
            labels_list = []
            save_dir = train_split if mode == 'train' else eval_split
            save_dir = f'datainfo/data_saved_origin/{dataset}/{save_dir}/'
            for batch in data_fn:
                images = batch[0]['image'].numpy()
                labels = batch[1].numpy()
                os.makedirs(save_dir, exist_ok=True)
                for image_ix in range(images.shape[0]):
                    image_path = '/'.join(save_dir.strip('/').split('/')
                                          [-2:]) + '/' + str(iter) + '.png'
                    cv2.imwrite(save_dir + str(iter) + '.png',
                                images[image_ix][:, :, ::-1])
                    labels_list.append(image_path + ' ' +
                                       str(labels[image_ix]))
                    if iter % 10 == 0:
                        print('now', dataset, mode, iter)
                    iter += 1
            open(save_dir.rstrip('/') + '_meta.list',
                 'w').write('\n'.join(labels_list) + '\n')
