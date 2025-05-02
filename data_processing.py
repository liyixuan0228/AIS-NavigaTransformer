"""
AIS数据处理
模块化结构说明：
1. 配置管理模块：日志配置、全局参数
2. 核心处理器模块：数据加载、预处理、特征工程全流程
3. 辅助工具模块：数据保存、统计报告
"""

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------
# 模块一：配置管理
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------

import os
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import logging
from tqdm import tqdm

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('ais_processing.log'),
        logging.StreamHandler()
    ]
)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 模块二：核心处理器
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------

class RobustAISProcessor:
    """AIS数据处理核心类，包含完整数据处理流水线"""
    
    def __init__(self, port_distance_threshold=10, max_time_gap_train=3, max_time_gap_test=3):
        """
        初始化处理器
        参数：
            port_distance_threshold (int): 港口距离判定阈值(公里)
            max_time_gap_train (float): 训练集最大允许时间间隔(小时)
            max_time_gap_test (float): 测试集最大允许时间间隔(小时)
        """
        # 处理参数设置
        self.port_distance_threshold = port_distance_threshold
        self.max_time_gap_train = max_time_gap_train
        self.max_time_gap_test = max_time_gap_test
        
        # 初始化组件
        self._init_scalers()      # 归一化器
        self._load_port_data()    # 港口数据
        self._init_stats()        # 统计信息

    # -------------------------------------------------------------------------------------------------
    # 初始化相关方法
    # -------------------------------------------------------------------------------------------------
    
    def _init_stats(self):
        """初始化轨迹统计信息容器"""
        self.train_stats = {
            'total_trajectories': 0,    # 总轨迹数
            'discarded_short': 0,       # 丢弃短轨迹数
            'lengths': [],              # 轨迹长度分布
            'min_length': np.inf,       # 最小轨迹长度
            'max_length': 0,            # 最大轨迹长度
            'failed_ships': []          # 处理失败船舶列表
        }
        self.test_stats = {  # 结构同train_stats
            'total_trajectories': 0,
            'discarded_short': 0,
            'lengths': [],
            'min_length': np.inf,
            'max_length': 0,
            'failed_ships': []
        }

    def _init_scalers(self):
        """初始化各特征的MinMax归一化器"""
        self.scalers = {
            'lon': MinMaxScaler(),      # 经度
            'lat': MinMaxScaler(),      # 纬度
            'sog': MinMaxScaler(),      # 航速
            'cog': MinMaxScaler(),      # 航向
            'hdg': MinMaxScaler(),      # 船艏向
            'draught': MinMaxScaler()   # 吃水深度
        }

    def _load_port_data(self):
        """
        加载港口参考数据
        支持格式：Excel > CSV，自动处理索引
        """
        port_file = 'ports.xls'  # 优先级1：Excel文件
        try:
            if os.path.exists(port_file):
                try:
                    self.ports = pd.read_excel(port_file, engine='openpyxl')
                except:  # 回退到xlrd引擎
                    self.ports = pd.read_excel(port_file, engine='xlrd')
                logging.info(f"成功加载Excel格式港口数据: {port_file}")
            elif os.path.exists('ports.csv'):  # 优先级2：CSV文件
                self.ports = pd.read_csv('ports.csv')
                logging.info("成功加载CSV格式港口数据")
            else:
                raise FileNotFoundError("未找到港口数据文件")
                
            self.ports = self.ports.set_index('port_code')  # 设置港口代码为索引
            logging.info(f"港口数据字段: {self.ports.columns.tolist()}")
            
        except Exception as e:
            logging.error(f"港口数据加载失败: {str(e)}")
            self.ports = pd.DataFrame(columns=['timezone_offset'])
            logging.warning("将使用默认时区偏移(0)继续处理")

    # ----------------------------------------------------------------------------------------------------
    # 主处理流程
    # ----------------------------------------------------------------------------------------------------
    
    def process(self, train_path, test_path):
        """
        执行完整处理流程
        参数：
            train_path (str): 训练数据文件路径
            test_path (str): 测试数据文件路径
        返回：
            tuple: (训练集, 验证集, 测试集)
        """
        try:
            # 训练数据处理
            train_data = self._process_dataset(train_path, is_train=True)
            # 划分验证集
            train, valid = train_test_split(train_data, test_size=0.2, random_state=42)
            # 测试数据处理
            test = self._process_dataset(test_path, is_train=False)
            # 持久化结果
            self._save_results(train, valid, test)
            return train, valid, test
            
        except Exception as e:
            logging.error(f"处理流程失败: {str(e)}")
            raise

    # ----------------------------------------------------------------------------------------------------
    # 数据集处理方法
    # ----------------------------------------------------------------------------------------------------
    
    def _process_dataset(self, file_path, is_train=True):
        """
        处理单个数据集
        参数：
            file_path (str): 数据文件路径
            is_train (bool): 是否为训练模式
        返回：
            list: 处理后的轨迹序列列表
        """
        logging.info(f"开始处理 {'训练' if is_train else '测试'} 数据: {file_path}")
        
        # 数据加载
        df = self._load_ais_data(file_path, is_train)
        # 预处理
        df = self._preprocess_data(df, is_train)
        # 航段划分
        df = self._segment_voyages(df, is_train)
        # 特征归一化
        df = self._normalize_features(df, is_train)
        # 序列构建
        return self._build_sequences(df, is_train)

    def _load_ais_data(self, file_path, is_train):
        """
        加载AIS数据并进行基础校验
        参数：
            file_path (str): 数据文件路径
            is_train (bool): 是否训练模式
        返回：
            pd.DataFrame: 有效数据
        """
        try:
            # 读取CSV并解析时间字段
            df = pd.read_csv(
                file_path,
                parse_dates=['slice_time'],                        # 自动解析时间戳
                dtype={
                    'ship_name': 'int32',                          # 优化存储类型
                    'status': 'str'                                # 保留原始字符串类型
                },
                date_parser=lambda x: pd.to_datetime(x, utc=True)  # UTC时间解析
            )
            
            # 字段完整性检查
            required_cols = ['ship_name', 'slice_time', 'lon', 'lat', 'status']
            if is_train:
                required_cols += ['leg_end_port_code']             # 训练集需要港口代码
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"缺少必要字段: {missing}")
                
            logging.info(f"成功加载数据: {len(df)} 条记录")
            return df.dropna(subset=['lon', 'lat'])                # 剔除空坐标
            
        except Exception as e:
            logging.error(f"数据加载失败: {str(e)}")
            raise

    # ----------------------------------------------------------------------------------------------------
    # 预处理方法组
    # ----------------------------------------------------------------------------------------------------
    
    def _preprocess_data(self, df, is_train):
        """
        数据预处理阶段
        包含时区转换、状态字段处理
        """
        try:
            if is_train:
                # 合并港口时区信息
                df = df.merge(
                    self.ports[['timezone_offset']],  # 选择时区偏移列
                    left_on='leg_end_port_code',      # 左表关联字段
                    right_index=True,                 # 右表使用索引
                    how='left'                        # 左连接
                )
                # 处理缺失时区偏移
                df['timezone_offset'] = df['timezone_offset'].fillna(0).astype('int16')
                # 转换为UTC时间
                df['utc_time'] = df['slice_time'].dt.tz_convert(None) - pd.to_timedelta(df['timezone_offset'], unit='h')
            else:
                # 测试集直接转换
                df['utc_time'] = df['slice_time'].dt.tz_localize(None)
            
            return df
        except TypeError as e:
            logging.error(f"时区转换错误: {str(e)}")
            logging.info("尝试备用时区转换方案...")
            # 备用方案：统一转换为无时区时间
            df['slice_time'] = pd.to_datetime(df['slice_time'], utc=True).dt.tz_convert(None)
            df['utc_time'] = df['slice_time']
            return df

    def _segment_voyages(self, df, is_train):
        """
        航段划分逻辑
        基于时间间隔、港口变更、状态转移多条件
        """
        logging.info("开始航段划分...")
        df = df.sort_values(['ship_name', 'utc_time'])    # 按船舶和时间排序
        
        # 计算时间间隔（小时）
        df['time_gap'] = df.groupby('ship_name')['utc_time'].diff().dt.total_seconds() / 3600
        
        split_points = pd.Series(False, index=df.index)   # 初始化分割点
        
        if is_train:
            # 训练集多条件分割
            df = self._merge_port_data(df)                # 合并港口坐标
            df = self._safe_convert_status(df)            # 状态字段处理
            
            # 定义分割条件
            cond_time = df['time_gap'] > self.max_time_gap_train       # 时间间隔条件
            cond_port_change = self._detect_port_change(df)            # 港口变更条件
            cond_status = self._detect_status_transition(df)           # 状态转移条件
            split_points = cond_time | cond_port_change | cond_status  # 逻辑或组合
        else:
            # 测试集仅时间条件
            cond_time = df['time_gap'] > self.max_time_gap_test
            split_points = cond_time
        
        # 生成航段ID
        df['voyage_id'] = (split_points.groupby(df['ship_name']).cumsum() + 1)
        logging.info(f"划分出 {df['voyage_id'].nunique()} 个航段")
        return df
    
    def _merge_port_data(self, df):
        """合并港口坐标信息并计算距离"""
        try:
            merged = df.merge(
                self.ports[['lon', 'lat']],   # 港口经纬度
                left_on='leg_end_port_code',  # 关联字段
                right_index=True,             # 右表使用索引
                how='left',                   # 左连接
                suffixes=('', '_port')        # 字段后缀
            )
            # 计算港口距离（公里）
            merged['port_distance'] = merged.apply(
                lambda row: geodesic(
                    (row['lat'], row['lon']),                                   # 当前坐标
                    (row.get('lat_port', np.nan), row.get('lon_port', np.nan))  # 港口坐标
                ).km if pd.notna(row.get('lat_port')) else np.inf,
                axis=1
            )
            return merged
        except Exception as e:
            logging.error(f"港口数据合并失败: {str(e)}")
            df['port_distance'] = np.inf                                        # 失败时设为无限远
            return df

    def _safe_convert_status(self, df):
        """安全转换状态字段"""
        # 转为数值型
        df['status'] = pd.to_numeric(df['status'], errors='coerce')
        # 前向填充缺失
        df['status'] = df.groupby('ship_name')['status'].ffill()
        # 填充剩余缺失为0（航行状态）
        df['status'] = df['status'].fillna(0).astype(int)
        # 验证有效状态码
        invalid_status = ~df['status'].isin([0, 1, 5])
        if invalid_status.any():
            logging.warning(f"发现{invalid_status.sum()}条无效状态记录,已重置为0")
            df.loc[invalid_status, 'status'] = 0
        return df

    def _detect_port_change(self, df):
        """检测目的港口变更"""
        return (
            (df['leg_end_port_code'] != df['leg_end_port_code'].shift(1))  # 港口代码变化
            & (df['ship_name'] == df['ship_name'].shift(1))                #  同一船舶
        )
        
    def _detect_status_transition(self, df):
        """检测有效到港状态转移"""
        arrival_cond = (df['status'] == 5) & (df['port_distance'] <= self.port_distance_threshold)
        status_change_cond = (df['status'] == 5) & (df['status'].shift(1) != 5)
        return arrival_cond & status_change_cond
    
    # ----------------------------------------------------------------------------------------------------
    # 特征工程方法
    # ----------------------------------------------------------------------------------------------------
    
    def _normalize_features(self, df, is_train):
        """特征归一化处理"""
        features = ['lon', 'lat', 'sog', 'cog', 'hdg', 'draught']
        
        if is_train:
            logging.info("训练模式: 拟合归一化参数")
            for feat in features:
                if len(df) > 0:
                    data = df[feat].values.reshape(-1, 1)
                    self.scalers[feat].fit(data) if len(data) > 0 else None
        
        # 应用归一化
        for feat in features:
            if len(df) > 0:
                df[feat] = self.scalers[feat].transform(df[feat].values.reshape(-1, 1)).flatten()
        return df

    # ----------------------------------------------------------------------------------------------------
    # 序列构建与结果保存
    # ----------------------------------------------------------------------------------------------------
    
    def _build_sequences(self, df, is_train):
        """构建轨迹序列"""
        sequences = []
        stats = self.train_stats if is_train else self.test_stats  # 选择统计容器
        
        grouped = df.groupby(['ship_name', 'voyage_id'])
        for (ship_id, voyage_id), group in tqdm(grouped, desc="构建轨迹"):
            try:
                # 过滤短轨迹
                if len(group) < 13:
                    stats['discarded_short'] += 1
                    continue
                
                # 记录统计信息
                seq_len = len(group)
                stats['lengths'].append(seq_len)
                stats['min_length'] = min(stats['min_length'], seq_len)
                stats['max_length'] = max(stats['max_length'], seq_len)
                stats['total_trajectories'] += 1
                
                # 构建序列数据
                sequences.append({
                    'ship_id': int(ship_id),
                    'voyage_id': int(voyage_id),
                    'features': group[['lon', 'lat', 'sog', 'cog', 'hdg', 'draught']].values,
                    'timestamps': group['utc_time'].values.astype('datetime64[s]'),
                    'raw_lon': group['lon'].values,  # 原始归一化后的经度
                    'raw_lat': group['lat'].values   # 原始归一化后的纬度
                })
                
            except Exception as e:
                stats['failed_ships'].append(ship_id)
                logging.error(f"轨迹构建失败: 船舶 {ship_id} 航段 {voyage_id} - {str(e)}")
        
        return sequences

    def _save_results(self, train, valid, test):
        """保存处理结果并生成报告"""
        # 数据序列化
        output = {
            'train': train,
            'valid': valid,
            'test': test,
            'scalers': self.scalers,         # 归一化器
            'train_stats': self.train_stats, # 训练统计
            'test_stats': self.test_stats    # 测试统计
        }
        
        # 分文件保存
        for name, data in output.items():
            with open(f'ais_processed_{name}.pkl', 'wb') as f:
                pickle.dump(data, f)
            logging.info(f"已保存 {name} 数据集 ({len(data) if isinstance(data, list) else '-'} 条)")
        
        # 生成数据处理文本报告
        report = [
            "=== 数据处理报告 ===",
            "【训练集统计】",
            f"总轨迹数: {self.train_stats['total_trajectories']}",
            f"轨迹长度范围: {self.train_stats['min_length']}-{self.train_stats['max_length']}",
            f"平均轨迹长度: {np.mean(self.train_stats['lengths']):.1f}",
            f"丢弃短轨迹: {self.train_stats['discarded_short']}",
            "【测试集统计】",
            f"总轨迹数: {self.test_stats['total_trajectories']}",
            f"轨迹长度范围: {self.test_stats['min_length']}-{self.test_stats['max_length']}",
            f"平均轨迹长度: {np.mean(self.test_stats['lengths']):.1f}",
            f"丢弃短轨迹: {self.test_stats['discarded_short']}"
        ]
        logging.info("\n".join(report))
        
        
"""
最后结果保存为以下序列 :                 
'ship_id''voyage_id''timestamps''raw_lon' 'raw_lat''lon', 'lat', 'sog', 'cog', 'hdg', 'draught'
其中：
'raw_lon' # 原始归一化后的经度
'raw_lat' # 原始归一化后的纬度
"""
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# 模块三：执行入口
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # 初始化处理器
    processor = RobustAISProcessor(
        port_distance_threshold=10,  # 港口距离阈值(公里)
        max_time_gap_train=3.0,      # 训练集时间间隔阈值
        max_time_gap_test=3.0        # 测试集时间间隔阈值
    )
    processor.process(train_path='ais_train.csv',test_path='ais_test.csv')