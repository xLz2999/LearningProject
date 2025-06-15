import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
import os
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.titlesize'] = 14
plt.rcParams["axes.unicode_minus"] = False

class GroundwaterDepthPredictor:
    def __init__(self, data_file="Dataset_processed.xlsx", station_name="Station1", 
                 seq_length=12, batch_size=32, epochs=500, output_dir="results"):
        self.data_file = data_file
        self.station_name = station_name
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_and_preprocess_data(self):
        """加载数据并进行预处理"""
        logger.info(f"加载数据: {self.data_file}, 站点: {self.station_name}")
        
        try:
            if not os.path.exists(self.data_file):
                logger.warning(f"数据文件不存在: {self.data_file}, 生成模拟数据")
                return self.generate_synthetic_data()
            
            df = pd.read_excel(self.data_file, sheet_name=self.station_name)
            logger.info(f"成功加载数据, 形状: {df.shape}")
            
            # 处理缺失值
            missing_ratio = df.isna().sum() / len(df)
            if missing_ratio.sum() > 0:
                logger.info(f"缺失值比例:\n{missing_ratio[missing_ratio > 0]}")
                df = df.interpolate(method='cubic')
                logger.info("缺失值处理完成")
            
            return df
        except Exception as e:
            logger.error(f"数据加载错误: {e}")
            return self.generate_synthetic_data()
    
    def generate_synthetic_data(self):
        """生成模拟数据用于测试"""
        logger.info("生成模拟数据")
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
            'Tem(℃)': temp,
            'Evaporation (万m³)': evaporation,
            'Depth (m)': depth
        })
        
        logger.info(f"模拟数据生成完成, 形状: {df.shape}")
        return df
    
    def create_sequences(self, df):
        """生成时间序列数据"""
        logger.info("创建时间序列数据")
        
        target_col = "Depth (m)"
        features = [col for col in df.columns if col not in [target_col, 'Date']]
        
        X, y = [], []
        for i in range(len(df) - self.seq_length):
            X.append(df[features].values[i:i+self.seq_length])
            y.append(df[target_col].values[i+self.seq_length])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"序列数据创建完成: X形状={X.shape}, y形状={y.shape}")
        return X, y, features
    
    def scale_data(self, X, y):
        """数据标准化"""
        logger.info("数据标准化")
        
        # 保存原始数据用于后续反归一化
        self.original_y = y.copy()
        
        # 特征标准化
        self.scaler_X = StandardScaler()
        X_scaled = self.scaler_X.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        
        # 目标标准化
        self.scaler_y = StandardScaler()
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        return X_scaled, y_scaled
    
    def train_test_split(self, X_scaled, y_scaled):
        """划分训练集和测试集"""
        logger.info("划分训练集和测试集")
        
        # 确保测试集是时间上的最后一部分
        test_size = int(0.2 * len(X_scaled))
        if test_size < 1:
            test_size = 1
        
        X_train, X_test = X_scaled[:-test_size], X_scaled[-test_size:]
        y_train, y_test = y_scaled[:-test_size], y_scaled[-test_size:]
        
        logger.info(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
        return X_train, X_test, y_train, y_test
    
    def create_data_loaders(self, X_train, y_train, X_test, y_test):
        """创建数据加载器"""
        logger.info("创建数据加载器")
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test).to(self.device)
        
        # 创建数据集
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, test_loader
    
    def build_model(self, input_features):
        """构建CNN模型"""
        logger.info("构建CNN模型")
        
        class CNNTemporalModel(nn.Module):
            def __init__(self, input_features, kernel_size=3, hidden_channels=32):
                super(CNNTemporalModel, self).__init__()
                # 两层卷积提取特征
                self.conv1 = nn.Conv1d(input_features, hidden_channels, kernel_size, padding=kernel_size//2)
                self.conv2 = nn.Conv1d(hidden_channels, hidden_channels*2, kernel_size, padding=kernel_size//2)
                
                # 激活函数与批归一化
                self.relu = nn.ReLU()
                self.bn1 = nn.BatchNorm1d(hidden_channels)
                self.bn2 = nn.BatchNorm1d(hidden_channels*2)
                
                # 全局平均池化与全连接层
                self.gap = nn.AdaptiveAvgPool1d(1)
                self.fc = nn.Linear(hidden_channels*2, 1)
                
            def forward(self, x):
                # 输入形状: (batch, seq_len, features) -> (batch, features, seq_len)
                x = x.permute(0, 2, 1)
                
                # 第一层卷积
                x = self.bn1(self.relu(self.conv1(x)))
                # 第二层卷积
                x = self.bn2(self.relu(self.conv2(x)))
                # 全局平均池化
                x = self.gap(x).squeeze(-1)
                # 全连接层输出
                return self.fc(x).squeeze(-1)
        
        model = CNNTemporalModel(input_features).to(self.device)
        logger.info(f"模型结构:\n{model}")
        return model
    
    def train(self, model, train_loader):
        """训练模型"""
        logger.info("开始训练模型")
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        loss_history = []
        for epoch in range(self.epochs):
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
                logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
        
        return loss_history
    
    def evaluate(self, model, test_loader):
        """评估模型"""
        logger.info("评估模型")
        
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                y_pred = model(X_batch)
                all_preds.append(y_pred.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
        
        # 合并所有批次的结果
        y_pred_scaled = np.concatenate(all_preds)
        y_true_scaled = np.concatenate(all_targets)
        
        # 反归一化
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = self.scaler_y.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
        
        return y_true, y_pred
    
    def calculate_metrics(self, y_true, y_pred):
        """计算评估指标"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        logger.info(f"评估结果（站点{self.station_name}）:")
        logger.info(f"RMSE: {rmse:.4f} m")
        logger.info(f"MAE: {mae:.4f} m")
        logger.info(f"R²: {r2:.4f}")
        
        return rmse, mae, r2
    
    def plot_loss(self, loss_history):
        """绘制损失函数曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history)
        plt.title('训练损失函数（MSE）变化')
        plt.xlabel('训练轮次（Epoch）')
        plt.ylabel('损失值')
        plt.grid(True)
        
        # 保存图像
        file_path = os.path.join(self.output_dir, f"{self.station_name}_loss_curve.png")
        plt.savefig(file_path)
        logger.info(f"保存损失曲线: {file_path}")
        plt.close()
    
    def plot_prediction(self, y_true, y_pred):
        """绘制预测值与实际值对比图"""
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label='实际埋深', c='blue', linewidth=2)
        plt.plot(y_pred, label='CNN预测值', c='red', linestyle='--', linewidth=2)
        plt.title(f'地下水埋深预测结果对比（站点{self.station_name}）', fontsize=14)
        plt.xlabel('时间序列索引', fontsize=12)
        plt.ylabel('埋深（m）', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 保存图像
        file_path = os.path.join(self.output_dir, f"{self.station_name}_pred_vs_actual.png")
        plt.savefig(file_path)
        logger.info(f"保存预测对比图: {file_path}")
        plt.close()
    
    def plot_scatter(self, y_true, y_pred, metrics):
        """绘制预测值与实际值散点图"""
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.6, c='green', s=50)
        
        # 添加对角线
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        # 添加统计信息
        rmse, mae, r2 = metrics
        stats_text = f'RMSE = {rmse:.4f}\nMAE = {mae:.4f}\nR² = {r2:.4f}'
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
                 fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.title(f'预测值与实际值散点图（站点{self.station_name}）', fontsize=14)
        plt.xlabel('实际埋深（m）', fontsize=12)
        plt.ylabel('预测埋深（m）', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 保存图像
        file_path = os.path.join(self.output_dir, f"{self.station_name}_scatter_plot.png")
        plt.savefig(file_path)
        logger.info(f"保存散点图: {file_path}")
        plt.close()
    
    def run(self):
        """运行整个流程"""
        start_time = datetime.now()
        logger.info(f"开始地下水埋深预测流程: {start_time}")
        
        try:
            # 1. 加载和预处理数据
            df = self.load_and_preprocess_data()
            
            # 2. 创建时间序列
            X, y, features = self.create_sequences(df)
            
            # 3. 数据标准化
            X_scaled, y_scaled = self.scale_data(X, y)
            
            # 4. 划分数据集
            X_train, X_test, y_train, y_test = self.train_test_split(X_scaled, y_scaled)
            
            # 5. 创建数据加载器
            train_loader, test_loader = self.create_data_loaders(X_train, y_train, X_test, y_test)
            
            # 6. 构建模型
            model = self.build_model(len(features))
            
            # 7. 训练模型
            loss_history = self.train(model, train_loader)
            
            # 8. 评估模型
            y_true, y_pred = self.evaluate(model, test_loader)
            
            # 9. 计算评估指标
            metrics = self.calculate_metrics(y_true, y_pred)
            
            # 10. 可视化结果
            self.plot_loss(loss_history)
            self.plot_prediction(y_true, y_pred)
            self.plot_scatter(y_true, y_pred, metrics)
            
            logger.info("预测流程完成")
            
        except Exception as e:
            logger.exception(f"预测流程出错: {e}")
        
        end_time = datetime.now()
        logger.info(f"总耗时: {end_time - start_time}")

if __name__ == "__main__":
    # 创建预测器实例
    predictor = GroundwaterDepthPredictor(
        data_file="Dataset_processed.xlsx",
        station_name="Station1",
        seq_length=12,
        batch_size=32,
        epochs=500,
        output_dir="results"
    )
    
    # 运行预测流程
    predictor.run()