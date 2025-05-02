"""
AIS轨迹预测系统 - 分区域Transformer预测
实现基于地理区域划分的多模型训练与预测——AIS-NavigaTransformer模型
"""
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 配置管理
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import os                                                                                                               # 操作系统接口
import torch                                                                                                            # PyTorch深度学习框架
import torch.nn as nn                                                                                                   # PyTorch神经网络模块
import torch.nn.functional as F                                                                                         # PyTorch函数接口
import torch.optim as optim                                                                                             # PyTorch优化器
from torch.utils.data import Dataset, DataLoader                                                                        # PyTorch数据加载工具
import numpy as np                                                                                                      # 数值计算库
import pickle                                                                                                           # Python对象序列化
import json                                                                                                             # JSON处理
import pandas as pd                                                                                                     # 数据分析库
from tqdm import tqdm                                                                                                   # 进度条工具
from pathlib import Path                                                                                                # 路径操作
import cartopy.crs as ccrs                                                                                              # 地理投影系统
import cartopy.feature as cfeature                                                                                      # 地理特征
from sklearn.preprocessing import MinMaxScaler                                                                          # 数据归一化
from sklearn.cluster import KMeans                                                                                      # K均值聚类
import matplotlib                                                                                                       # 绘图库
matplotlib.use('Agg')                                                                                                   # 使用非交互式后端
import matplotlib.pyplot as plt                                                                                         # 绘图接口

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 轨迹区域划分模块
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class RegionManager:
    def __init__(self, n_regions=10):
        """初始化区域管理器"""
        self.n_regions = n_regions                                                      # 区域数量
        self.region_models = {}                                                         # 存储每个区域的模型
        self.region_scalers = {}                                                        # 每个区域的归一化器
        self.kmeans = None                                                              # 区域聚类模型
        self.region_centers = None                                                      # 区域中心点
    
    def fit_regions(self, trajectories):
        """
        根据轨迹数据聚类划分区域
        参数:
            trajectories: 轨迹数据列表，每个元素是包含'features'的字典
        """
        # 提取所有轨迹点的经纬度
        all_points = np.vstack([traj['features'][:, :2] for traj in trajectories])
        
        # 使用KMeans进行区域划分
        self.kmeans = KMeans(n_clusters=self.n_regions, random_state=42)
        self.kmeans.fit(all_points)
        self.region_centers = self.kmeans.cluster_centers_
        
        # 为每个区域创建归一化器
        for region_id in range(self.n_regions):
            region_data = []
            for traj in trajectories:
                # 获取属于当前区域的点
                labels = self.kmeans.predict(traj['features'][:, :2])
                region_points = traj['features'][labels == region_id]
                if len(region_points) > 0:
                    region_data.append(region_points)
            
            if len(region_data) > 0:
                region_data = np.vstack(region_data)
                scaler = MinMaxScaler()
                scaler.fit(region_data)
                self.region_scalers[region_id] = scaler
    
    def get_region_id(self, point):
        """
        获取点的所属区域ID
        参数:
            point: 包含经纬度的点坐标
        返回:
            区域ID
        """
        if self.kmeans is None:
            raise ValueError("RegionManager not fitted yet")
        return self.kmeans.predict([point[:2]])[0]
    
    def normalize_for_region(self, data, region_id):
        """
        使用指定区域的归一化器进行归一化
        参数:
            data: 待归一化数据
            region_id: 区域ID
        返回:
            归一化后的数据
        """
        if region_id not in self.region_scalers:
            raise ValueError(f"Region {region_id} not found")
        return self.region_scalers[region_id].transform(data)
    
    def denormalize_for_region(self, data, region_id):
        """
        使用指定区域的反归一化
        参数:
            data: 待反归一化数据
            region_id: 区域ID
        返回:
            反归一化后的原始数据
        """
        if region_id not in self.region_scalers:
            raise ValueError(f"Region {region_id} not found")
        return self.region_scalers[region_id].inverse_transform(data)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 配置参数类 
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 256                                                                 # 批大小
        self.max_epochs = 50                                                                  # 最大训练轮数
        self.learning_rate = 3e-4                                                             # 学习率
        self.patience = 10                                                                     # 早停耐心值
        self.max_seqlen = 935                                                                 # 最大序列长度，与数据预处理一致
        self.min_seqlen = 13                                                                  # 最小序列长度，与数据预处理一致
        self.init_seqlen = 36                                                                 # 初始观测点数量
        self.n_embd = 1024                                                                     # 嵌入维度
        self.n_head = 12                                                                       # 注意力头数
        self.n_layer = 8                                                                      # Transformer层数
        self.embd_pdrop = 0.1                                                                 # 嵌入层dropout率
        self.attn_pdrop = 0.1                                                                 # 注意力dropout率
        self.resid_pdrop = 0.1                                                                # 残差连接dropout率
        self.scaled_features = ['lon', 'lat', 'sog', 'cog', 'hdg', 'draught']                 # 特征
        self.full_size = len(self.scaled_features)                                            # 特征维度
        self.n_regions = 10                                                                   # 划分区域数量
        self.data_dir = Path(".")                                                             # 数据目录
        self.result_dir = Path("./training_results")                                          # 结果目录
        self._create_dirs()                                                                   # 创建目录结构
        self.region_manager = None                                                            # 区域管理器，将在数据加载后初始化
    
    def _create_dirs(self):
        self.result_dir.mkdir(exist_ok=True)
        (self.result_dir/"test_vis").mkdir(exist_ok=True)                                     # 轨迹预测可视化存储
        (self.result_dir/"loss_curves").mkdir(exist_ok=True)                                  # 存储训练损失曲线
        (self.result_dir/"region_models").mkdir(exist_ok=True)                                # 存储各区域模型参数

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 数据集类 
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class AISDataset(Dataset):
    def __init__(self, data_path, config, region_id=None):
        """
        参数:
            data_path: 数据文件路径
            config: 配置对象
            region_id: 如果为None则使用全局数据，否则只使用指定区域的数据
        """
        self.config = config
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)                                           # 加载预处理后的轨迹列表
        
        # 初始化区域管理器
        if self.config.region_manager is None:
            self.config.region_manager = RegionManager(n_regions=config.n_regions)
            self.config.region_manager.fit_regions(self.data)
        
        # 如果指定了区域，则过滤数据
        if region_id is not None:
            self.data = self._filter_by_region(region_id)
    
    def _filter_by_region(self, region_id):
        """
        过滤出包含指定区域点的轨迹
        参数:
            region_id: 要过滤的区域ID
        返回:
            过滤后的轨迹列表
        """
        filtered_data = []
        for traj in self.data:
            # 获取轨迹点的区域标签
            labels = self.config.region_manager.kmeans.predict(traj['features'][:, :2])
            if region_id in labels:
                filtered_data.append(traj)
        return filtered_data
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        返回:
            seq: 特征序列 (max_seqlen, full_size)
            mask: 序列掩码 (max_seqlen)
            region_labels: 区域标签 (max_seqlen)
            seq_len: 实际序列长度
            ship_id: 船舶ID
            voyage_id: 航次ID
        """
        traj = self.data[idx]
        seq = np.zeros((self.config.max_seqlen, self.config.full_size))
        seq_len = min(len(traj['features']), self.config.max_seqlen)
        seq[:seq_len] = traj['features'][:seq_len]                     # 自动填充短序列
        mask = np.zeros(self.config.max_seqlen)
        mask[:seq_len] = 1                                             # 有效位置标记
        
        # 添加区域信息
        region_labels = np.zeros(self.config.max_seqlen)
        if seq_len > 0:
            points = seq[:seq_len, :2]
            region_labels[:seq_len] = self.config.region_manager.kmeans.predict(points)
        
        return (
            torch.FloatTensor(seq),                                    # 特征序列
            torch.FloatTensor(mask),                                   # 掩码
            torch.LongTensor(region_labels),                           # 区域标签
            seq_len,                                                   # 实际长度
            traj['ship_id'],                                           # 船舶ID
            traj['voyage_id']                                          # 航次ID
        )

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Transformer架构
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class TrAISformer(nn.Module):
    def __init__(self, config):
        """
        AIS轨迹预测Transformer模型
        参数:
            config: 配置对象
        """
        super().__init__()
        self.config = config
        
        # 输入嵌入层
        self.input_emb = nn.Linear(config.full_size, config.n_embd)
        
        # 位置编码
        self.pos_emb = nn.Parameter(torch.zeros(1, config.max_seqlen, config.n_embd))
        
        # Transformer模块
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        
        # 输出层
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.full_size)
        
        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
                
    def forward(self, x):
        """
        前向传播
        参数:
            x: 输入序列 (batch_size, seq_len, input_dim)
        返回:
            预测序列 (batch_size, seq_len, input_dim)
        """
        B, T, _ = x.shape
        
        # 特征嵌入
        tok_emb = self.input_emb(x)
        
        # 位置编码
        pos_emb = self.pos_emb[:, :T, :]
        x = tok_emb + pos_emb
        
        # Transformer块
        for block in self.blocks:
            x = block(x)
            
        # 输出预测
        x = self.ln_f(x)
        return self.head(x)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        """
        Transformer块
        参数:
            config: 配置对象
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        """前向传播"""
        x = x + self.attn(self.ln1(x))                   # 残差连接+多头注意力
        x = x + self.mlp(self.ln2(x))                    # 残差连接+前馈网络
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        """
        多头注意力机制
        参数:
            config: 配置对象
        """
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        
        # QKV投影
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        
        # Dropout
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        
        # 因果掩码
        self.register_buffer("mask", torch.tril(torch.ones(config.max_seqlen, config.max_seqlen)))

    def forward(self, x):
        """
        前向传播
        参数:
            x: 输入张量 (batch_size, seq_len, n_embd)
        返回:
            注意力输出 (batch_size, seq_len, n_embd)
        """
        B, T, C = x.size()
        
        # 生成QKV
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力
        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        
        # 合并注意力头
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 模型训练流程 
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class Trainer:
    def __init__(self, config):
        """训练器类
        参数:
            config: 配置对象
        """
        self.config = config
        self._init_datasets()                     # 初始化数据集
        self._init_models()                       # 初始化模型
        # 为每个区域创建优化器和学习率调度器
        self.optimizers = {
            region_id: optim.AdamW(model.parameters(), lr=config.learning_rate)
            for region_id, model in self.models.items()
        }
        self.schedulers = {
            region_id: optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
            for region_id, optimizer in self.optimizers.items()
        }
        # 训练状态跟踪
        self.best_losses = {region_id: float('inf') for region_id in range(config.n_regions)}
        self.early_stop_counters = {region_id: 0 for region_id in range(config.n_regions)}
        self.train_losses = {region_id: [] for region_id in range(config.n_regions)}
        self.valid_losses = {region_id: [] for region_id in range(config.n_regions)}
    
    def _init_datasets(self):
        """初始化数据加载器"""
        # 全局数据用于区域划分
        with open("ais_processed_train.pkl", "rb") as f:
            train_data = pickle.load(f)
        
        # 初始化区域管理器
        self.config.region_manager = RegionManager(n_regions=self.config.n_regions)
        self.config.region_manager.fit_regions(train_data)
        
        # 为每个区域创建数据加载器
        self.train_loaders = {}
        self.valid_loaders = {}
        self.test_loaders = {}
        
        for region_id in range(self.config.n_regions):
            self.train_loaders[region_id] = DataLoader(
                AISDataset("ais_processed_train.pkl", self.config, region_id),
                batch_size=self.config.batch_size,
                shuffle=True,
                pin_memory=True
            )
            self.valid_loaders[region_id] = DataLoader(
                AISDataset("ais_processed_valid.pkl", self.config, region_id),
                batch_size=self.config.batch_size,
                pin_memory=True
            )
        
        # 测试集使用全局数据
        self.test_loader = DataLoader(
            AISDataset("ais_processed_test.pkl", self.config),
            batch_size=self.config.batch_size,
            pin_memory=True
        )
    
    def _init_models(self):
        """为每个区域初始化模型"""
        self.models = {
            region_id: TrAISformer(self.config).to(self.config.device)
            for region_id in range(self.config.n_regions)
        }
    
    def train_region_epoch(self, region_id):
        """
        训练单个区域的一个epoch
        参数:
            region_id: 区域ID
        返回:
            平均训练损失
        """
        self.models[region_id].train()
        total_loss = 0.0
        loader = self.train_loaders[region_id]
        
        progress_bar = tqdm(loader, desc=f"Training Region {region_id}", leave=False)
        for batch in progress_bar:
            x, mask, *_ = batch
            x = x.to(self.config.device, non_blocking=True)
            mask = mask.to(self.config.device, non_blocking=True)
            
            self.optimizers[region_id].zero_grad(set_to_none=True)
            
            # 前向传播
            outputs = self.models[region_id](x)
            
            # 计算掩码损失
            loss = F.mse_loss(outputs[:, :-1], x[:, 1:], reduction='none')
            loss = (loss * mask[:, 1:].unsqueeze(-1)).sum() / mask[:, 1:].sum()
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.models[region_id].parameters(), 1.0)
            self.optimizers[region_id].step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        return total_loss / len(loader)
    
    def validate_region(self, region_id):
        """
        验证单个区域
        参数:
            region_id: 区域ID
        返回:
            平均验证损失
        """
        self.models[region_id].eval()
        total_loss = 0.0
        loader = self.valid_loaders[region_id]
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Validating Region {region_id}", leave=False):
                x, mask, *_ = batch
                x = x.to(self.config.device, non_blocking=True)
                
                outputs = self.models[region_id](x)
                loss = F.mse_loss(outputs[:, :-1], x[:, 1:], reduction='none')
                loss = (loss * mask[:, 1:].unsqueeze(-1)).sum() / mask[:, 1:].sum()
                total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def _denormalize(self, data, region_id):
        """
        使用指定区域的归一化器进行反归一化
        参数:
            data: 待反归一化数据
            region_id: 区域ID
        返回:
            反归一化后的数据
        """
        original_shape = data.shape
        # 将数据reshape为2D (samples, features)
        data_2d = data.reshape(-1, data.shape[-1])
        denorm = self.config.region_manager.denormalize_for_region(data_2d, region_id)
        # 恢复原始形状
        return denorm.reshape(original_shape)
    
    def _plot_loss_curve(self, epoch):
        """
        绘制所有区域的损失曲线
        参数:
            epoch: 当前epoch数
        """
        plt.figure(figsize=(12, 8))
        for region_id in range(self.config.n_regions):
            if len(self.train_losses[region_id]) > 0:
                plt.plot(self.train_losses[region_id], label=f'Region {region_id} Train')
                plt.plot(self.valid_losses[region_id], '--', label=f'Region {region_id} Valid')
        
        plt.xlabel('Epoch')
        plt.ylabel('LCE Loss')
        plt.title('Training Progress by Region')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.config.result_dir/f"loss_curves/epoch_{epoch+1}.png")
        plt.close()
    
    def _plot_region_map(self):
        """绘制区域划分地图"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        
        # 添加地图要素
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, facecolor='azure')
        ax.add_feature(cfeature.COASTLINE, edgecolor='gray')
        ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='darkgray')
        ax.gridlines(color='gray', linestyle='--')
        
        # 绘制区域中心点
        centers = self.config.region_manager.region_centers
        ax.scatter(
            centers[:, 0], centers[:, 1],
            transform=ccrs.PlateCarree(),
            c=range(len(centers)), cmap='tab20',
            s=100, edgecolors='black', linewidths=0.5,
            label='Region Centers'
        )
        
        # 添加区域编号
        for i, (lon, lat) in enumerate(centers):
            ax.text(lon, lat, str(i), transform=ccrs.PlateCarree(),
                    ha='center', va='center', fontweight='bold')
        
        plt.title("Region Division Map")
        plt.legend()
        plt.savefig(self.config.result_dir/"region_map.png", bbox_inches='tight')
        plt.close()
    
    def _plot_test_trajectory(self, input_seq, pred_seq, ship_id, voyage_id):
        """轨迹地理可视化（解决多线程问题）
        参数:
            input_seq: 输入序列
            pred_seq: 预测序列
            ship_id: 船舶ID
            voyage_id: 航次ID
        """
        # 创建图形对象时指定使用Agg backend
        with plt.ioff():  # 关闭交互模式
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            
            # 添加地图要素
            ax.add_feature(cfeature.LAND, facecolor='lightgray')
            ax.add_feature(cfeature.OCEAN, facecolor='azure')
            ax.add_feature(cfeature.COASTLINE, edgecolor='gray')
            ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='darkgray')
            ax.gridlines(color='gray', linestyle='--')
            
            # 绘制轨迹
            if len(input_seq) > 0:
                ax.plot(
                    input_seq[:, 0], input_seq[:, 1],
                    transform=ccrs.PlateCarree(),
                    marker='o', color='blue', markersize=5,
                    linewidth=1.5, label='Actual'
                )
            
            if len(pred_seq) > 0:
                ax.plot(
                    pred_seq[:, 0], pred_seq[:, 1],
                    transform=ccrs.PlateCarree(),
                    linestyle='--', marker='s', color='orange', markersize=4,
                    linewidth=1.5, label='Predicted'
                )
                
                # 绘制连接线
                if len(input_seq) > 0 and len(pred_seq) > 0:
                    ax.plot(
                        [input_seq[-1, 0], pred_seq[0, 0]],
                        [input_seq[-1, 1], pred_seq[0, 1]],
                        transform=ccrs.PlateCarree(),
                        linestyle='--', color='green', alpha=0.7,
                        linewidth=1, label='Transition'
                    )
            
            plt.title(f"Ship {ship_id} | Voyage {voyage_id}", pad=20)
            plt.legend(loc='upper left')
            
            # 保存并立即关闭图形
            plt.savefig(self.config.result_dir/f"test_vis/{ship_id}_{voyage_id}.png", 
                    bbox_inches='tight', dpi=150)
            plt.close(fig)                                 # 关键：显式关闭图形释放资源
    
    def _predict_in_region(self, model, initial_seq, region_id, max_steps):
        """
        在指定区域内进行预测
        参数:
            model: 预测模型
            initial_seq: 初始序列
            region_id: 区域ID
            max_steps: 最大预测步数
        返回:
            预测序列
        """
        model.eval()
        current_seq = initial_seq.to(self.config.device)
        predictions = []
        
        with torch.no_grad():
            for _ in range(max_steps):
                # 预测下一步
                next_step = model(current_seq)[:, -1:]
                predictions.append(next_step)
                
                # 更新序列
                current_seq = torch.cat([current_seq, next_step], dim=1)[:, -self.config.max_seqlen:]
                
                # 检查是否还在当前区域
                last_point = next_step[0, -1, :2].cpu().numpy()
                current_region = self.config.region_manager.get_region_id(last_point)
                if current_region != region_id:
                    break
        
        if len(predictions) > 0:
            return torch.cat(predictions, dim=1)
        return None
    
    def test(self):
        """完整测试流程 - 多区域预测版本"""
        # 加载各区域最佳模型
        for region_id in range(self.config.n_regions):
            model_path = self.config.result_dir/f"region_models/best_model_region_{region_id}.pt"
            if os.path.exists(model_path):
                self.models[region_id].load_state_dict(torch.load(model_path))
        
        results = []
        predictions = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                x, mask, region_labels, seq_lens, ship_ids, voyage_ids = batch
                
                for i in range(x.size(0)):
                    # 获取初始序列和区域
                    input_seq = x[i, :self.config.init_seqlen].unsqueeze(0)
                    current_region = region_labels[i, self.config.init_seqlen-1].item()
                    
                    # 逐步预测
                    full_prediction = []
                    remaining_steps = self.config.max_seqlen - self.config.init_seqlen
                    current_input = input_seq
                    
                    while remaining_steps > 0 and current_region is not None:
                        # 在当前区域预测
                        pred = self._predict_in_region(
                            self.models[current_region],
                            current_input,
                            current_region,
                            remaining_steps
                        )
                        
                        if pred is None or len(pred) == 0:
                            break
                        
                        # 添加到完整预测
                        full_prediction.append(pred.cpu())
                        remaining_steps -= pred.size(1)
                        
                        # 更新当前输入和区域
                        last_point = pred[0, -1, :2].cpu().numpy()
                        current_region = self.config.region_manager.get_region_id(last_point)
                        current_input = torch.cat([current_input, pred], dim=1)[:, -self.config.max_seqlen:]
                    
                    # 处理预测结果
                    if len(full_prediction) > 0:
                        full_prediction = torch.cat(full_prediction, dim=1)
                        #input_denorm = self._denormalize(input_seq.numpy(), region_labels[i, 0].item())
                        input_denorm = self._denormalize(input_seq.numpy(), region_labels[i, 0].item())
                        pred_denorm = []
                        
                        # 反归一化每个区域的部分
                        start_idx = 0
                        current_region = region_labels[i, self.config.init_seqlen-1].item()
                        for j in range(full_prediction.size(1)):
                            point = full_prediction[0, j, :2].numpy()
                            region = self.config.region_manager.get_region_id(point)
                            
                            if region != current_region or j == full_prediction.size(1)-1:
                                # 反归一化当前区域的部分
                                region_slice = full_prediction[:, start_idx:j+1, :].numpy()
                                # 确保是2D数组 (samples, features)
                                region_slice_2d = region_slice.reshape(-1, region_slice.shape[-1])
                                denorm_slice = self.config.region_manager.denormalize_for_region(
                                    region_slice_2d, current_region
                                )
                                # 恢复原始形状
                                denorm_slice = denorm_slice.reshape(region_slice.shape)
                                pred_denorm.append(denorm_slice[0])
                                start_idx = j+1
                                current_region = region
                        
                        if len(pred_denorm) > 0:
                            pred_denorm = np.vstack(pred_denorm)
                            
                            # 构建时间序列
                            base_time = pd.Timestamp.now().floor('H')
                            timestamps = [base_time + pd.Timedelta(hours=3*j) 
                                        for j in range(self.config.max_seqlen)]
                            
                            # 保存结果
                            record = {
                                'ship_id': ship_ids[i].item(),
                                'voyage_id': voyage_ids[i].item(),
                                'timestamps': [str(ts) for ts in timestamps],
                                'input': input_denorm[0].tolist(),
                                'prediction': pred_denorm.tolist()
                            }
                            results.append(record)
                            
                            # 保存为CSV数据
                            for j, ts in enumerate(timestamps[:self.config.init_seqlen]):
                                predictions.append({
                                    **{feat: input_denorm[0,j,k] for k, feat in enumerate(self.config.scaled_features)},
                                    'ship_id': ship_ids[i].item(),
                                    'voyage_id': voyage_ids[i].item(),
                                    'timestamp': ts.isoformat(),
                                    'is_prediction': 0,
                                    'region': region_labels[i, j].item()
                                })
                            
                            for j, ts in enumerate(timestamps[self.config.init_seqlen:self.config.init_seqlen+len(pred_denorm)]):
                                point_region = self.config.region_manager.get_region_id(pred_denorm[j, :2])
                                predictions.append({
                                    **{feat: pred_denorm[j,k] for k, feat in enumerate(self.config.scaled_features)},
                                    'ship_id': ship_ids[i].item(),
                                    'voyage_id': voyage_ids[i].item(),
                                    'timestamp': ts.isoformat(),
                                    'is_prediction': 1,
                                    'region': point_region
                                })
                            
                            # 轨迹可视化
                            self._plot_test_trajectory(
                                input_denorm[0], 
                                pred_denorm, 
                                ship_ids[i].item(),
                                voyage_ids[i].item()
                            )
        
        # 保存结果
        pd.DataFrame(predictions).to_csv(self.config.result_dir/"predictions.csv", index=False)
        with open(self.config.result_dir/"predictions.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"测试完成，结果保存在 {self.config.result_dir}")
    
    def run(self):
        """完整训练流程 - 多区域版本"""
        # 绘制区域划分图
        self._plot_region_map()
        
        for epoch in range(self.config.max_epochs):
            # 训练和验证每个区域
            for region_id in range(self.config.n_regions):
                # 跳过没有数据的区域
                if len(self.train_loaders[region_id]) == 0:
                    continue
                
                # 训练
                train_loss = self.train_region_epoch(region_id)
                self.train_losses[region_id].append(train_loss)
                
                # 验证
                valid_loss = self.validate_region(region_id)
                self.valid_losses[region_id].append(valid_loss)
                
                # 早停机制
                if valid_loss < self.best_losses[region_id]:
                    self.best_losses[region_id] = valid_loss
                    torch.save(
                        self.models[region_id].state_dict(),
                        self.config.result_dir/f"region_models/best_model_region_{region_id}.pt"
                    )
                    self.early_stop_counters[region_id] = 0
                else:
                    self.early_stop_counters[region_id] += 1
                
                # 学习率调整
                self.schedulers[region_id].step(valid_loss)
            
            # 绘制损失曲线
            self._plot_loss_curve(epoch)
            
            # 打印进度
            print(f"Epoch {epoch+1}/{self.config.max_epochs}")
            for region_id in range(self.config.n_regions):
                if len(self.train_losses[region_id]) > 0:
                    print(f"Region {region_id} - Train Loss: {self.train_losses[region_id][-1]:.4f} | Valid Loss: {self.valid_losses[region_id][-1]:.4f}")
            
            # 检查是否所有区域都早停
            if all(counter >= self.config.patience for counter in self.early_stop_counters.values()):
                print("All regions early stopped")
                break
        
        # 最终测试
        self.test()

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 主程序
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # 配置并行线程
    torch.set_num_threads(4)
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['TK_SILENCE_DEPRECATION'] = '1'
    
    try:
        config = Config()
        trainer = Trainer(config)
        trainer.run()
    except Exception as e:
        print(f"程序异常终止: {str(e)}")
        # 显式清理资源
        if 'trainer' in locals():
            del trainer
        torch.cuda.empty_cache()
        exit(1)