import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import BasicDataset  # 引入自定义数据集
from net import SVAnet  # 引入定义的去噪网络
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 自定义损失函数
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.mse_loss = nn.MSELoss()  # 使用均方误差作为基础损失函数

    def forward(self, prediction, target, input_data):
        prediction_mean = torch.mean(prediction)  # 计算预测值的均值
        diff_from_mean = torch.abs(prediction - prediction_mean)  # 计算与均值的差异

        # 根据差异选择损失函数
        loss = torch.where(diff_from_mean < 0.3,
                           self.mse_loss(prediction, target),
                           self.mse_loss(prediction, input_data))
        return loss.mean()  # 返回标量损失

# 配置训练参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 256  # 根据 GPU 显存调整批量大小
epochs = 300  # 训练轮数
learning_rate = 1e-3  # 学习率
train_data_path = 'data/train'  # 训练数据路径
val_data_path = 'data/val'  # 验证数据路径
model_save_dir = 'models'  # 模型保存目录
os.makedirs(model_save_dir, exist_ok=True)  # 确保目录存在

# 加载数据集
train_dataset = BasicDataset(is_train=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

val_dataset = BasicDataset(is_train=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

# 初始化模型
model = SVAnet().to(device)
criterion = CustomLoss()  # 定义损失函数
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 使用Adam优化器

# 训练过程
def train_one_epoch(epoch):
    model.train()  # 切换到训练模式
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training")):
        inputs, targets = inputs.to(device), targets.to(device)  # 将数据移动到设备上
        optimizer.zero_grad()  # 清空梯度
        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, targets, inputs)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 优化模型参数
        running_loss += loss.item()  # 累加损失
        if (i + 1) % 10 == 0:
            logger.info(f"Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    avg_loss = running_loss / len(train_loader)  # 计算平均损失
    logger.info(f"Epoch [{epoch+1}/{epochs}], Average Training Loss: {avg_loss:.4f}")
    return avg_loss

# 验证过程
def validate():
    model.eval()  # 切换到验证模式
    running_loss = 0.0
    with torch.no_grad():  # 禁用梯度计算
        for inputs, targets in tqdm(val_loader, desc="Validation"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets, inputs)  # 计算损失
            running_loss += loss.item()
    avg_loss = running_loss / len(val_loader)  # 计算平均损失
    logger.info(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss

# 主程序保护
if __name__ == '__main__':
    best_val_loss = float('inf')
    for epoch in range(epochs):
        train_loss = train_one_epoch(epoch)  # 进行一轮训练
        val_loss = validate()  # 进行验证

        # 每5轮保存一次模型
        if (epoch + 1) % 5 == 0:
            model_epoch_path = os.path.join(model_save_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_epoch_path)
            logger.info(f"Model saved at epoch {epoch+1}")

        # 如果验证损失更小，保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(model_save_dir, 'best_denoising_model.pth')
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Validation loss improved, best model saved at epoch {epoch+1}")

    # 保存最终模型
    final_model_path = os.path.join(model_save_dir, 'final_denoising_model.pth')
    torch.save(model.state_dict(), final_model_path)
    logger.info("Training complete, final model saved.")
