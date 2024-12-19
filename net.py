import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class WeightGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        self.reshape_net = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),  # 使用 5x5 卷积核，padding=2 保证输出长度相同
            nn.BatchNorm1d(64),
            Mish(),  # 使用 Mish 激活函数
        )

    def forward(self, x):
        x = self.reshape_net(x)
        return x


# 定义一维信号去噪的 CNN 网络
class DenoisingCNN(nn.Module):
    def __init__(self):
        super(DenoisingCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=5, padding=2),  # 使用 5x5 卷积核，padding=2 保证输出长度相同
            nn.BatchNorm1d(128),
            Mish(),  # 使用 Mish 激活函数
            nn.MaxPool1d(2),  # 下采样，长度减半
            nn.Conv1d(128, 256, kernel_size=5, padding=2),  # 使用 5x5 卷积核，padding=2 保证输出长度相同
            nn.BatchNorm1d(256),
            Mish(),  # 使用 Mish 激活函数
            nn.MaxPool1d(2)  # 再次下采样
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),  # 使用 5x5 卷积核
            nn.BatchNorm1d(128),
            Mish(),  # 使用 Mish 激活函数
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),  # 恢复到输入大小
            nn.BatchNorm1d(64),
            Mish(),  # 使用 Mish 激活函数
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class SVAnet(nn.Module):
    def __init__(self):
        super(SVAnet, self).__init__()

        self.filterGenerator = DenoisingCNN()
        self.weightGenerator = WeightGenerator()

        # 定义用于输出的卷积层，生成最终结果
        self.output_conv = nn.Conv1d(64, 1, kernel_size=5, padding=2)  # 使用 5x5 卷积核

    def forward(self, x):
        # 传入数据到 filterGenerator 和 weightGenerator
        sv_filters = self.filterGenerator(x)  # 输入数据 x 到 filterGenerator
        weighted_feature_maps = self.weightGenerator(x)  # 输入数据 x 到 weightGenerator

        # 进行加权操作（两者的形状已保证一致，不再需要填充）
        filtered_result = sv_filters * weighted_feature_maps

        # 使用卷积层生成最终输出
        out_result = self.output_conv(filtered_result)

        # 输出进行归一化
        return F.normalize(out_result, p=float('inf'), dim=2, eps=1e-12, out=None)


# 测试模型形状
if __name__ == "__main__":
    # 创建网络实例
    model = SVAnet()

    # 创建一个虚拟输入数据，假设输入是批量大小为 1，信号长度为 256 的单通道数据
    input_tensor = torch.randn(1, 1, 256)  # (batch_size, channels, signal_length)

    # 打印输入数据的形状
    print("输入数据形状：", input_tensor.shape)

    # 通过模型进行前向传播
    output_tensor = model(input_tensor)

    # 打印输出数据的形状
    print("输出数据形状：", output_tensor.shape)
