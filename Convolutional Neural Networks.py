import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import os

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.titlesize'] = 14  # 修正标题字体大小
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 假设数据文件路径（请根据实际路径修改）
DATA_FILE = "Dataset_processed.xlsx"
STATION_NAME = "Station1"  

# 1. 数据加载与预处理
def load_data(file_path, station_name):
    """加载数据并处理缺失值"""
    try:
        df = pd.read_excel(file_path, sheet_name=station_name)
        print(f"成功加载{station_name}数据，形状：{df.shape}")
        # 检查缺失值
        missing_ratio = df.isna().sum() / len(df)
        print(f"缺失值比例：\n{missing_ratio[missing_ratio > 0]}")
        # 三次样条插值处理缺失值
        df = df.interpolate(method='cubic')
        print("缺失值处理完成")
        return df
    except Exception as e:
        print(f"数据加载错误：{e}")
        # 生成模拟数据用于测试
        np.random.seed(42)
        time = pd.date_range(start='2000-01-01', end='2023-12-01', freq='M')
        n_samples = len(time)
        irrigation = np.random.normal(50, 10, n_samples)
        rainfall = np.random.normal(30, 8, n_samples)
        temp = np.random.normal(15, 5, n_samples)
        evaporation = np.random.normal(20, 6, n_samples)
        # 模拟地下水埋深（受各因素影响）
        depth = 10 - 0.01 * irrigation - 0.02 * rainfall + 0.1 * temp + 0.05 * evaporation
        depth += np.random.normal(0, 0.5, n_samples)  # 加入噪声
        
        df = pd.DataFrame({
            'Date': time,
            'Irrigation(万m³)': irrigation,
            'Rainfall(万m³)': rainfall,
            'Temp(℃)': temp,
            'Evaporation(万m³)': evaporation,
            'Depth(m)': depth
        })
        print(f"使用模拟数据，形状：{df.shape}")
        return df

def create_sequences(data, seq_length, target_col):
    """生成滑动窗口序列"""
    features = [col for col in data.columns if col not in [target_col, 'Date']]
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[features].values[i:i+seq_length])
        y.append(data[target_col].values[i+seq_length])
    return np.array(X), np.array(y)

# 2. 模型定义 - 修正维度处理逻辑
class CNNTemporalModel(nn.Module):
    def __init__(self, input_features, kernel_size=3, hidden_channels=32):
        super(CNNTemporalModel, self).__init__()
        # 两层卷积提取特征 - 注意输入通道数应为1（单变量时间序列处理）
        self.conv1 = nn.Conv1d(1, hidden_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels*2, kernel_size, padding=kernel_size//2)
        # 激活函数与批归一化
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels*2)
        # 全局平均池化与全连接层
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_channels*2, 1)
        
    def forward(self, x):
        # 输入形状：(batch, seq_len, features) → 调整为(batch, 1, seq_len)
        batch, seq_len, features = x.shape
        
        x = x.reshape(batch, seq_len, features)  # 确保输入形状正确
        x = x.permute(0, 2, 1)  # 转换为(batch, features, seq_len)
        
        if features <= seq_len:
            x = x.reshape(batch, 1, seq_len * features)  # (batch, 1, seq_len*features)
        else:
            x = x[:, 0, :].unsqueeze(1)  # 仅使用第一个特征，(batch, 1, seq_len)
        
        # 第一层卷积
        x = self.bn1(self.relu(self.conv1(x)))
        # 第二层卷积
        x = self.bn2(self.relu(self.conv2(x)))
        # 全局平均池化
        x = self.gap(x).squeeze(-1)
        # 全连接层输出
        return self.fc(x).squeeze(-1)

# 3. 训练与评估
def train_model(model, train_loader, optimizer, criterion, epochs=500):
    """训练模型并记录损失"""
    loss_history = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    return loss_history

def evaluate_model(model, X_test, y_test, scaler_y):
    """评估模型并返回预测结果"""
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        y_pred_scaled = model(X_test_tensor).numpy()
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    return y_true, y_pred

# 4. 可视化结果
def plot_loss(loss_history):
    """绘制损失函数曲线"""
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title('训练损失函数（MSE）变化')
    plt.xlabel('训练轮次（Epoch）')
    plt.ylabel('损失值')
    plt.grid(True)
    plt.savefig('cnn_loss_curve.png')
    plt.show()

def plot_prediction(y_true, y_pred, title):
    """绘制预测值与实际值对比图"""
    plt.figure(figsize=(12, 6))
    # 绘制所有样本
    plt.plot(y_true, label='实际埋深', c='blue', linewidth=2)
    plt.plot(y_pred, label='CNN预测值', c='red', linestyle='--', linewidth=2)
    plt.title(title, fontsize=14)
    plt.xlabel('时间序列索引', fontsize=12)
    plt.ylabel('埋深（m）', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig('cnn_pred_vs_actual.png')
    plt.show()

    # 绘制散点图
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, c='green', s=30)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    plt.title('预测值与实际值散点图', fontsize=14)
    plt.xlabel('实际埋深（m）', fontsize=12)
    plt.ylabel('预测埋深（m）', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig('cnn_scatter_plot.png')
    plt.show()

# 主函数
if __name__ == "__main__":
    # 确保数据文件存在，否则使用模拟数据
    if not os.path.exists(DATA_FILE):
        print(f"警告：未找到数据文件{DATA_FILE}，使用模拟数据进行测试")
    
    # 1. 加载与预处理数据
    df = load_data(DATA_FILE, STATION_NAME)
    target_col = "Depth(m)"
    seq_length = 12  # 使用前12个月数据预测下一个月
    
    X, y = create_sequences(df, seq_length, target_col)
    print(f"生成序列数据：X形状={X.shape}, y形状={y.shape}")
    
    # 特征标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # 划分训练集与测试集（8:2）
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    print(f"训练集大小：{X_train.shape[0]}, 测试集大小：{X_test.shape[0]}")
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 2. 初始化模型
    input_features = X.shape[-1]  # 特征维度
    model = CNNTemporalModel(input_features=input_features)
    print(f"模型结构：{model}")
    
    # 定义损失函数与优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 3. 训练模型
    print("开始训练模型...")
    loss_history = train_model(model, train_loader, optimizer, criterion, epochs=500)
    
    # 4. 评估模型
    print("模型评估中...")
    y_true, y_pred = evaluate_model(model, X_test, y_test, scaler_y)
    
    # 5. 计算评估指标
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"评估结果（站点{STATION_NAME[-1]}）:")
    print(f"RMSE: {rmse:.4f} m, MAE: {mae:.4f} m")
    
    # 6. 可视化结果
    plot_loss(loss_history)
    plot_prediction(y_true, y_pred, f'地下水埋深预测结果对比（站点{STATION_NAME[-1]}）')