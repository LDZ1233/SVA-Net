import torch
import scipy.io
import numpy as np
from net import SVAnet  # 引入之前训练好的去噪模型

# 加载数据
def load_mat_data(file_path):
    # 从 .mat 文件中加载数据
    data = scipy.io.loadmat(file_path)
    # 假设数据是存储在名为 'mat' 的字段中，调整为实际数据字段名
    complex_signal = data['mat']
    return complex_signal


# 分解复数信号为实部和虚部
def split_real_imag(complex_signal):
    real_part = complex_signal.real
    imag_part = complex_signal.imag
    return real_part, imag_part


# 将实部和虚部合并成复数形式
def combine_real_imag(real_part, imag_part):
    return real_part + 1j * imag_part


# 按行批量处理信号
def process_by_rows(model, real_part, imag_part, device):
    # 将数据转换为 tensor，并移到设备上
    real_part_tensor = torch.tensor(real_part, dtype=torch.float32).unsqueeze(1).to(device)  # (256, 1, 256)
    imag_part_tensor = torch.tensor(imag_part, dtype=torch.float32).unsqueeze(1).to(device)

    # 通过模型批量处理每一行
    with torch.no_grad():
        real_output = model(real_part_tensor)  # 处理实部
        imag_output = model(imag_part_tensor)  # 处理虚部

    # 转回 NumPy 数组并返回
    real_output = real_output.squeeze().cpu().numpy()  # 去除多余维度
    imag_output = imag_output.squeeze().cpu().numpy()
    return real_output, imag_output


# 按列批量处理信号
def process_by_columns(model, real_part, imag_part, device):
    # 将数据转换为 tensor，并移到设备上
    real_part_tensor = torch.tensor(real_part.T, dtype=torch.float32).unsqueeze(1).to(device)  # 转置后处理列 (256, 1, 256)
    imag_part_tensor = torch.tensor(imag_part.T, dtype=torch.float32).unsqueeze(1).to(device)

    # 通过模型批量处理每一列
    with torch.no_grad():
        real_output = model(real_part_tensor)  # 处理实部
        imag_output = model(imag_part_tensor)  # 处理虚部

    # 转回 NumPy 数组并返回
    real_output = real_output.squeeze().cpu().numpy().T  # 转置回原始形状
    imag_output = imag_output.squeeze().cpu().numpy().T
    return real_output, imag_output


# 测试模型
def test_model(model, file_path, device, process_type='rows'):
    # 加载 .mat 数据
    complex_signal = load_mat_data(file_path)

    # 分解为实部和虚部
    real_part, imag_part = split_real_imag(complex_signal)

    if process_type == 'rows':
        real_output, imag_output = process_by_rows(model, real_part, imag_part, device)
    elif process_type == 'columns':
        real_output, imag_output = process_by_columns(model, real_part, imag_part, device)
    else:
        raise ValueError("Invalid process_type. Choose 'rows' or 'columns'.")

    # 合并处理结果
    denoised_signal = combine_real_imag(real_output, imag_output)  # 合并

    return denoised_signal


# 保存去噪信号到 .mat 文件
def save_denoised_signal(denoised_signal, output_path):
    # 将复数信号直接保存为一个复数数据
    scipy.io.savemat(output_path, {'denoised_signal': denoised_signal})
    print(f"去噪信号已保存至 {output_path}")


# 设置设备和加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SVAnet().to(device)
model.load_state_dict(torch.load('models/best_denoising_model.pth'))  # 加载最佳模型

# 测试模型并保存结果
file_path = 'exp/a1.mat'  # 输入文件路径

# 1. 使用行批处理并保存为 '1.mat'
denoised_signal_rows = test_model(model, file_path, device, process_type='rows')
output_path_rows = 'exp/1-1.mat'  # 保存路径
save_denoised_signal(denoised_signal_rows, output_path_rows)

# 2. 使用列批处理并保存为 '2.mat'
denoised_signal_columns = test_model(model, file_path, device, process_type='columns')
output_path_columns = 'exp/1-2.mat'  # 保存路径
save_denoised_signal(denoised_signal_columns, output_path_columns)
