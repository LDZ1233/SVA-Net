import scipy.io as scio
import os
import torch
from torch.utils.data import Dataset, DataLoader

class BasicDataset(Dataset):
    def __init__(self, is_train=True):
        super(BasicDataset, self).__init__()

        # 根据is_train选择训练集或验证集路径
        if is_train:
            self.sample_path = 'data/train/sample/'
            self.gt_path = 'data/train/gt/'
        else:
            self.sample_path = 'data/val/sample/'
            self.gt_path = 'data/val/gt/'

        # 获取样本数量
        self.length = len(os.listdir(self.sample_path))
        # 生成文件名列表
        self.file_name = [str(x) + '.mat' for x in range(self.length)]

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        file_name = self.file_name[i]
        sample_path = self.sample_path + file_name
        gt_path = self.gt_path + file_name

        # 加载.mat文件
        sample_mat = scio.loadmat(sample_path)
        sample = sample_mat['selected_sample_data']  # 假设数据在'mat'字段中

        gt_mat = scio.loadmat(gt_path)
        gt = gt_mat['selected_gt_data']  # 假设数据在'filtered_data'字段中

        # 仅提取实部，确保是单通道数据
        sample_real = sample.real  # 获取实部
        gt_real = gt.real  # 获取实部

        # 转换为torch张量，保持为单通道
        sample_tensor = torch.from_numpy(sample_real).float()  # 添加一个维度，变为 1xHxW
        gt_tensor = torch.from_numpy(gt_real).float()  # 添加一个维度，变为 1xHxW

        sample_tensor = sample_tensor.permute(1, 0)
        gt_tensor = gt_tensor.permute(1, 0)
        return sample_tensor, gt_tensor

if __name__ == '__main__':
    # 创建数据集实例
    Dataset = BasicDataset(is_train=True)
    val_loader = DataLoader(Dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    # 打印第一个样本的输入和目标张量大小
    print(Dataset[0][0].size())  # 打印第一个样本的输入大小
    print(Dataset[0][1].size())  # 打印第一个样本的目标大小
    print(len(Dataset))  # 打印数据集样本数量

    # 打印每个批次中的数据大小
    for item in val_loader:
        print(item[0].size())  # 打印输入数据的大小
        break
