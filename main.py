import argparse
import random
from datetime import datetime
import torch.utils.data as data

from datasets_operations import *
from utils import *

from train_test import *
from sklearn.metrics import classification_report
from results_report import metrics
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
import os
from model import ARSNet
import time
from thop import profile
from scipy.io import savemat
datasets_file = {
    'PU': ['paviaU.mat', 'paviaU_7gt.mat'],
    'PC': ['paviaC.mat', 'paviaC_7gt.mat'],
    'D': ['Dioni.mat', 'Dioni_gt_out68.mat'],
    'L': ['Loukia.mat', 'Loukia_gt_out68.mat'],
    'H13': ['Houston13.mat', 'Houston13_7gt.mat'],
    'H18': ['Houston18.mat', 'Houston18_7gt.mat'],
    # 'HC':['WHU_Hi_HanChuan.mat','WHU_Hi_HanChuan_gt.mat'],
    # 'HH':['WHU_Hi_HongHu.mat','WHU_Hi_HongHu_gt.mat']
}


parser = argparse.ArgumentParser(description='important parameters')
parser.add_argument('--save_path', type=str, default="results",
                    help='the path to save the model')
parser.add_argument('--data_path', type=str, default=r'datasets/',
                    help='the path to load the data')
parser.add_argument('--model_results', type=str, default=r"ARSNet/",
                    help='the path to save the model')

parser.add_argument('--pretrained_model_path', type=str, default=None,
                    help='If there is a pre-trained model, this is set to its path')
parser.add_argument('--cuda', type=int, default=0,
                    help="Specify CUDA device (defaults to -1, which learns on CPU)")  # CPU:-1 || GPU:0

# parser.add_argument('--dataset_name', type=str, default='Houston',
#                     help='Task dataset name')
# source_data, source_label = datasets_file['H13']
# target_data, target_label = datasets_file['H18']

# parser.add_argument('--dataset_name', type=str, default='Pavia',
#                     help='Task dataset name')
# source_data, source_label = datasets_file['PU']
# target_data, target_label = datasets_file['PC']

parser.add_argument('--dataset_name', type=str, default='Hyrank',
                    help='Task dataset name')
source_data, source_label = datasets_file['D']
target_data, target_label = datasets_file['L']

# parser.add_argument('--dataset_name', type=str, default='Wuhan',
#                     help='Task dataset name')
# source_data, source_label = datasets_file['HH']
# target_data, target_label = datasets_file['HC']

parser.add_argument('--source_data', type=str, default=source_data,
                    help='the name of the source data file')
parser.add_argument('--source_label', type=str, default=source_label,
                    help='the name of the source label file')
parser.add_argument('--target_data', type=str, default=target_data,
                    help='the name of the test data file')
parser.add_argument('--target_label', type=str, default=target_label,
                    help='the name of the test label file')

parser.add_argument('--patch_size', type=int, default=12, help="Size of the spatial neighbourhood (optional, if "
                    "absent will be set by the model)")
parser.add_argument('--lr', type=float, default=1e-2, help="Learning rate, set by the model if not specified.")
parser.add_argument('--batch_size', type=int, default=256,
                    help="Batch size (optional, if absent will be set by the model")
parser.add_argument('--seed', type=int, default=555, metavar='S',
                    help='random seed ')

parser.add_argument('--log_interval', type=int, default=4, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num_epoch', type=int, default=300,
                    help='the number of epoch')
parser.add_argument('--num_trials', type=int, default=1,
                    help='the number of epoch')
parser.add_argument('--training_sample_ratio', type=float, default=0.8,
                    help='fraction of labeled source pixels used for training; the remainder is source validation')
parser.add_argument('--re_ratio', type=int, default=3,
                    help='multiple of of data augmentation')

# Data augmentation parameters
group_da = parser.add_argument_group('Data augmentation')
group_da.add_argument('--flip_augmentation', action='store_true', default=True,
                      help="Random flips (if patch_size > 1)")
group_da.add_argument('--radiation_augmentation', action='store_true', default=True,
                      help="Random radiation noise (illumination)")
group_da.add_argument('--mixture_augmentation', action='store_true', default=False,
                      help="Random mixes between spectra")


args = parser.parse_args()
DEVICE = get_device(args.cuda)

def save_prediction_as_mat(predict_list, test_dataset, original_target_shape, filename):
    """
    将预测结果保存为.mat文件，仅包含一个名为'map'的字段
    
    Args:
        predict_list: test函数返回的预测列表
        test_dataset: 测试数据集对象
        original_target_shape: 原始目标图像的形状（不含padding）
        filename: 保存的.mat文件路径
    """
    
    # 合并所有预测结果
    all_predictions = np.concatenate(predict_list)
    
    # 创建预测图像（与原始目标图像形状相同，不含padding）
    prediction_map = np.zeros(original_target_shape, dtype=np.uint8)
    
    # 获取测试数据的位置索引
    test_indices = test_dataset.indices
    
    # 将预测结果填入对应位置
    # 注意：需要考虑padding的影响，将坐标转换回原始图像坐标
    r = test_dataset.patch_size // 2  # padding大小
    for i, (x, y) in enumerate(test_indices):
        if i < len(all_predictions):
            # 减去padding偏移量，得到原始坐标
            orig_x = x - r
            orig_y = y - r
            # 确保坐标在有效范围内
            if 0 <= orig_x < original_target_shape[0] and 0 <= orig_y < original_target_shape[1]:
                prediction_map[orig_x, orig_y] = all_predictions[i] + 1  # 转换回原始标签范围(1-N)
    
    # 保存为.mat文件，仅包含'map'字段
    savemat(filename, {'map': prediction_map})
    print(f"Prediction map saved to {filename}")
    
    return prediction_map

if __name__ == '__main__':
    seed_worker(args.seed)  # args.seed

    source_hsi, source_gt, target_hsi, target_gt, ignored_labels = get_dataset(args)

    sample_num_src = len(np.nonzero(source_gt)[0])
    sample_num_tar = len(np.nonzero(target_gt)[0])

    num_classes = int(source_gt.max())
    N_BANDS = source_hsi.shape[-1]
    hyperparams = vars(args)
    hyperparams.update({'n_classes': num_classes, 'n_bands': N_BANDS, 'ignored_labels': ignored_labels,
                        'device': DEVICE, 'center_pixel': False, 'supervision': 'full', 'seed': args.seed})

    r = int(hyperparams['patch_size'] / 2)

    source_hsi = np.pad(source_hsi, ((r, r), (r, r), (0, 0)), 'symmetric')
    target_hsi = np.pad(target_hsi, ((r, r), (r, r), (0, 0)), 'symmetric')

    source_gt = np.pad(source_gt, ((r, r), (r, r)), 'constant', constant_values=(0, 0))
    target_gt = np.pad(target_gt, ((r, r), (r, r)), 'constant', constant_values=(0, 0))

    # Split only the labeled source scene.  The validation split, rather than target
    # OA, is used to select the checkpoint.
    train_gt_src, val_gt_src, _, _ = sample_gt(source_gt, args.training_sample_ratio, mode='random')
    test_gt_tar, _, _, _ = sample_gt(target_gt, 1, mode='random')
    source_hsi_re, train_gt_src_re = source_hsi, train_gt_src

    for i in range(args.re_ratio - 1):
        source_hsi_re = np.concatenate((source_hsi_re, source_hsi))
        train_gt_src_re = np.concatenate(
            (train_gt_src_re, train_gt_src))

    train_dataset = HyperX(source_hsi_re, train_gt_src_re, **hyperparams)
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=hyperparams['batch_size'],
                                   pin_memory=True,
                                   worker_init_fn=seed_worker,
                                   generator=g,
                                   shuffle=True,
                                   drop_last=True)

    # Never augment validation or target evaluation samples.
    eval_hyperparams = hyperparams.copy()
    eval_hyperparams.update({
        'flip_augmentation': False,
        'radiation_augmentation': False,
        'mixture_augmentation': False,
    })
    val_dataset = HyperX(source_hsi, val_gt_src, **eval_hyperparams)
    val_loader = data.DataLoader(val_dataset,
                                 pin_memory=True,
                                 batch_size=hyperparams['batch_size'],
                                 shuffle=False,
                                 drop_last=False)
    test_dataset = HyperX(target_hsi, test_gt_tar, **eval_hyperparams)
    test_loader = data.DataLoader(test_dataset,
                                  pin_memory=True,
                                  # worker_init_fn=seed_worker,
                                  # generator=g,
                                  batch_size=hyperparams['batch_size'],
                                  drop_last=False)
    len_src_loader = len(train_loader)
    len_src_dataset = len(train_loader.dataset)
    len_val_dataset = len(val_loader.dataset)
    len_tar_dataset = len(test_loader.dataset)
    len_tar_loader = len(test_loader)

    hyperparams.update({
        'Number of source training samples': len_src_dataset,
        'Number of source validation samples': len_val_dataset,
        'Number of target testing samples': len_tar_dataset,
        'Checkpoint selection': 'source validation OA',
        'Target max OA': 'monitoring only; never used for checkpoint selection',
    })

    model = ARSNet(num_classes, N_BANDS, hyperparams['patch_size']).to(DEVICE)
    model_dict = model.state_dict()
    if args.pretrained_model_path and os.path.exists(args.pretrained_model_path):
        pretrained_dict = torch.load(args.pretrained_model_path, map_location=DEVICE).state_dict()
        model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params / (1024):.2f}K training parameters.')
    root_path = os.path.join(args.save_path)
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    task_logit_dir = os.path.join(root_path, args.dataset_name, args.model_results, time_str)
    if not os.path.exists(task_logit_dir):
        os.makedirs(task_logit_dir)
    with open(os.path.join(task_logit_dir, 'hyperparams.txt'), 'a') as file:
        for k, v in hyperparams.items():
            print(f'{k}:{v}')
            file.write(f'{k}:{v}\n')
        file.write(f'{total_trainable_params / (1024):.2f}K training parameters.\n')

    best_val_accuracy = -float('inf')
    best_target_accuracy_at_source_checkpoint = None
    max_target_accuracy = -float('inf')
    epoch_times = []
    for epoch in range(1, args.num_epoch + 1):
        epoch_start_time = time.time()
        model, _ = train(args, epoch, train_loader, len_src_dataset, DEVICE, model)
        epoch_end_time = time.time()
        val_accuracy, _, _ = test(model, val_loader, DEVICE, task_logit_dir, split_name='source validation')
        target_accuracy, predict_list, label_list = test(
            model, test_loader, DEVICE, task_logit_dir, split_name='target evaluation')
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)

        # Kept solely to reproduce the legacy monitoring convention.  It must not
        # affect checkpoint selection, hyperparameters, or the saved prediction map.
        if target_accuracy > max_target_accuracy:
            max_target_accuracy = target_accuracy
            with open(os.path.join(task_logit_dir, 'log_accuracy.txt'), 'a') as file:
                file.write(f'epoch {epoch}: target monitoring max OA = {max_target_accuracy:.2f}%\n')

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_target_accuracy_at_source_checkpoint = target_accuracy
            predict_all = np.concatenate(predict_list)
            label_all = np.concatenate(label_list)

            results = metrics(predict_all, label_all,
                            ignored_labels=hyperparams['ignored_labels'],
                            n_classes=num_classes)

            logs_report = classification_report(label_all, predict_all,
                                                target_names=[str(i) for i in range(1, num_classes + 1)],digits=4)
            print(logs_report)

            checkpoint_log = (
                f'source-validation-selected checkpoint: epoch {epoch}, '
                f'source validation OA: {best_val_accuracy:.2f}%, '
                f'target OA at this checkpoint: {target_accuracy:.2f}%\n{logs_report}\n'
            )
            print(checkpoint_log, end='')

            with open(os.path.join(task_logit_dir, 'log_accuracy.txt'), 'a') as file:
                file.write(checkpoint_log)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'source_validation_accuracy': val_accuracy,
                'target_accuracy_at_source_selected_checkpoint': target_accuracy,
            }, os.path.join(task_logit_dir, f'best_model_epoch.pt'))
            original_target_shape = (target_gt.shape[0] - 2*r, target_gt.shape[1] - 2*r)
            mat_filename = os.path.join(task_logit_dir, f'prediction_map.mat')
            prediction_map = save_prediction_as_mat(predict_list, test_dataset, original_target_shape, mat_filename)

        else:
            print(
                f'source validation OA: {val_accuracy:.2f}% | '
                f'best source validation OA: {best_val_accuracy:.2f}% | '
                f'target OA (monitoring only): {target_accuracy:.2f}%\n'
            )

    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    total_training_time = sum(epoch_times)
    time_stats = f'\nTraining time statistics:\n' \
                    f'Total epochs: {len(epoch_times)}\n' \
                    f'Total training time: {total_training_time:.2f} seconds\n' \
                    f'Average time per epoch: {avg_epoch_time:.2f} seconds\n'
    with open(os.path.join(task_logit_dir, 'log_accuracy.txt'), 'a') as file:
            file.write(time_stats)
            file.write(
                f'final source-validation-selected target OA: '
                f'{best_target_accuracy_at_source_checkpoint:.2f}%\n'
            )
            file.write(f'legacy target monitoring max OA: {max_target_accuracy:.2f}%\n')
