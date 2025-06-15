import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import seaborn as sns

# 设置绘图风格 - 修复样式问题
plt.style.use('seaborn-v0_8-whitegrid')  # 使用兼容的样式名称
sns.set_style("whitegrid")

# 1. 数据加载和预处理
def load_data(sheet_name):
    # 从Excel加载指定sheet的数据
    file_path = 'Dataset_processed.xlsx'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件未找到: {file_path}")
    
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    # 检查数据列是否存在
    required_columns = ['Irrigation(万m³)', 'Rainfall(万m³)', 'Tem(℃)', 'Evaporation (万m³)', 'Depth (m)']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"数据中缺少必要的列: {col}")
    
    # 选择特征和目标列
    features = df[['Irrigation(万m³)', 'Rainfall(万m³)', 'Tem(℃)', 'Evaporation (万m³)']]
    target = df[['Depth (m)']]
    
    # 归一化处理
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    scaled_features = feature_scaler.fit_transform(features)
    scaled_target = target_scaler.fit_transform(target)
    
    return scaled_features, scaled_target, feature_scaler, target_scaler, df

# 2. 创建时间序列数据集
def create_sequences(features, target, seq_length):
    if len(features) != len(target):
        raise ValueError("特征和目标数组长度不一致")
    
    if seq_length >= len(features):
        raise ValueError("序列长度大于数据集长度")
    
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:i+seq_length])
        y.append(target[i+seq_length])
    return np.array(X), np.array(y)

# 3. 定义PyTorch数据集
class DepthDataset(Dataset):
    def __init__(self, features, targets):
        if len(features) != len(targets):
            raise ValueError("特征和目标数量不一致")
        
        self.X = features
        self.y = targets
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), \
               torch.tensor(self.y[idx], dtype=torch.float32)

# 4. 定义LSTM模型
class LSTMDepthPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        device = x.device
        batch_size = x.size(0)
        
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 5. 训练函数（返回损失历史）
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=500):
    model.train()
    loss_history = []
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        epoch_loss = total_loss/len(train_loader)
        loss_history.append(epoch_loss)
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}')
    
    return loss_history

# 绘制损失曲线
def plot_loss_curve(loss_history, station_name):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, 'b-', linewidth=2)
    plt.title(f'{station_name} - Training Loss Curve (MSE)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 确保输出目录存在
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/{station_name}_loss_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

# 绘制预测结果对比图
def plot_predictions_vs_actual(predictions, actuals, station_name):
    plt.figure(figsize=(12, 6))
    
    # 仅显示前50个点以便更清晰
    n_points = min(50, len(predictions))
    indices = np.arange(n_points)
    
    plt.plot(indices, actuals[:n_points], 'o-', color='#1f77b4', linewidth=2, 
             markersize=6, label='Actual Depth', alpha=0.9)
    plt.plot(indices, predictions[:n_points], 's-', color='#ff7f0e', linewidth=2, 
             markersize=5, label='LSTM Prediction', alpha=0.8)
    
    plt.title(f'{station_name} - Groundwater Depth Prediction', fontsize=14)
    plt.xlabel('Time Series Index', fontsize=12)
    plt.ylabel('Depth (m)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 确保输出目录存在
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/{station_name}_pred_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.close()

# 绘制预测值与实际值散点图
def plot_scatter(predictions, actuals, station_name):
    plt.figure(figsize=(8, 8))
    
    # 计算R²和RMSE
    r2 = r2_score(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    
    # 创建散点图
    plt.scatter(actuals, predictions, c='#2ca02c', alpha=0.6, s=50, 
                edgecolor='w', linewidth=0.5)
    
    # 添加对角线
    min_val = min(np.min(actuals), np.min(predictions)) * 0.95
    max_val = max(np.max(actuals), np.max(predictions)) * 1.05
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    # 添加统计信息
    plt.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}', 
             transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.title(f'{station_name} - Prediction vs Actual', fontsize=14)
    plt.xlabel('Actual Depth (m)', fontsize=12)
    plt.ylabel('Predicted Depth (m)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')
    plt.tight_layout()
    
    # 确保输出目录存在
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/{station_name}_scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

# 主程序
def main(station_name, sequence_length=12):
    try:
        print(f"处理站点: {station_name}")
        
        # 加载数据
        features, target, feature_scaler, target_scaler, df = load_data(station_name)
        print(f"数据集大小: {len(df)} 条记录")
        
        # 创建时间序列数据集
        X, y = create_sequences(features, target, sequence_length)
        print(f"创建的时间序列样本数: {len(X)}")
        
        # 划分训练集和测试集
        test_size = int(0.2 * len(X))
        if test_size < 1:
            test_size = 1
        
        X_train, X_test = X[:-test_size], X[-test_size:]
        y_train, y_test = y[:-test_size], y[-test_size:]
        
        print(f"训练集大小: {len(X_train)}")
        print(f"测试集大小: {len(X_test)}")
        
        # 创建数据加载器
        train_dataset = DepthDataset(X_train, y_train)
        test_dataset = DepthDataset(X_test, y_test)
        
        batch_size = min(16, len(X_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        # 初始化模型
        input_size = X_train.shape[2]  # 特征数量
        hidden_size = 64
        num_layers = 2
        output_size = 1
        
        model = LSTMDepthPredictor(input_size, hidden_size, num_layers, output_size).to(device)
        print(f"模型结构: {model}")
        
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 训练模型并获取损失历史
        loss_history = train_model(model, train_loader, criterion, optimizer, device, num_epochs=500)
        
        # 绘制损失曲线
        plot_loss_curve(loss_history, station_name)
        
        # 评估模型
        model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(labels.cpu().numpy())
        
        # 反归一化
        predictions = target_scaler.inverse_transform(np.array(predictions))
        actuals = target_scaler.inverse_transform(np.array(actuals))
        
        # 计算评估指标
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        r2 = r2_score(actuals, predictions)
        print(f"测试结果 - RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        # 绘制预测结果对比图
        plot_predictions_vs_actual(predictions.flatten(), actuals.flatten(), station_name)
        
        # 绘制散点图
        plot_scatter(predictions.flatten(), actuals.flatten(), station_name)
        
        print(f"{station_name} 处理完成，图表已保存到 plots/ 目录\n")
        return model
        
    except Exception as e:
        print(f"处理站点 {station_name} 时出错: {str(e)}")
        return None

# 运行主程序
if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs('plots', exist_ok=True)
    
    # 分别对两个站点进行训练和预测
    station1_model = main('Station1')
    
    station2_model = main('Station2')
    
    print("程序执行完成")