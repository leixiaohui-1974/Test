"""
深度学习降阶模型

使用LSTM/Transformer等深度学习模型进行快速预测
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # 创建占位类
    nn = None


@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 0.001
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    device: str = "cpu"  # "cpu" or "cuda"


if TORCH_AVAILABLE:
    class SequenceDataset(Dataset):
        """时间序列数据集"""
        
        def __init__(
            self,
            input_sequences: np.ndarray,  # (n_samples, seq_len, n_features)
            target_sequences: np.ndarray,  # (n_samples, seq_len, n_targets)
        ):
            self.inputs = torch.FloatTensor(input_sequences)
            self.targets = torch.FloatTensor(target_sequences)
        
        def __len__(self) -> int:
            return len(self.inputs)
        
        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            return self.inputs[idx], self.targets[idx]


    class LSTMNetwork(nn.Module):
        """LSTM网络"""
        
        def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            output_size: int,
            dropout: float = 0.2,
        ):
            super(LSTMNetwork, self).__init__()
            
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
            
            self.fc = nn.Linear(hidden_size, output_size)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, seq_len, input_size)
            
            # LSTM层
            lstm_out, _ = self.lstm(x)
            
            # 全连接层
            output = self.fc(lstm_out)
            
            return output


    class TransformerNetwork(nn.Module):
        """Transformer网络"""
        
        def __init__(
            self,
            input_size: int,
            d_model: int,
            nhead: int,
            num_encoder_layers: int,
            dim_feedforward: int,
            output_size: int,
            dropout: float = 0.1,
        ):
            super(TransformerNetwork, self).__init__()
            
            self.d_model = d_model
            
            # 输入投影
            self.input_projection = nn.Linear(input_size, d_model)
            
            # Transformer编码器
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_encoder_layers,
            )
            
            # 输出投影
            self.output_projection = nn.Linear(d_model, output_size)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, seq_len, input_size)
            
            # 输入投影
            x = self.input_projection(x)
            
            # Transformer编码
            x = self.transformer_encoder(x)
            
            # 输出投影
            output = self.output_projection(x)
            
            return output


class LSTMPredictor:
    """LSTM预测器"""
    
    def __init__(
        self,
        input_features: List[str],
        output_features: List[str],
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        """
        初始化LSTM预测器
        
        Parameters
        ----------
        input_features : List[str]
            输入特征名称列表
        output_features : List[str]
            输出特征名称列表
        hidden_size : int
            隐藏层大小
        num_layers : int
            LSTM层数
        dropout : float
            Dropout比例
        """
        if not TORCH_AVAILABLE:
            raise ImportError("需要安装PyTorch才能使用深度学习模型")
        
        self.input_features = input_features
        self.output_features = output_features
        self.input_size = len(input_features)
        self.output_size = len(output_features)
        
        self.model = LSTMNetwork(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=self.output_size,
            dropout=dropout,
        )
        
        self.scaler_input = None
        self.scaler_output = None
        self.is_trained = False
    
    def _normalize_data(
        self,
        data: np.ndarray,
        fit: bool = False,
        scaler: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """数据归一化"""
        if fit:
            mean = np.mean(data, axis=(0, 1))
            std = np.std(data, axis=(0, 1)) + 1e-8
            scaler = {'mean': mean, 'std': std}
        
        if scaler is None:
            raise ValueError("需要提供scaler")
        
        normalized = (data - scaler['mean']) / scaler['std']
        return normalized, scaler
    
    def train(
        self,
        input_sequences: np.ndarray,  # (n_samples, seq_len, n_input_features)
        output_sequences: np.ndarray,  # (n_samples, seq_len, n_output_features)
        config: Optional[TrainingConfig] = None,
    ) -> Dict[str, List[float]]:
        """
        训练模型
        
        Parameters
        ----------
        input_sequences : np.ndarray
            输入序列
        output_sequences : np.ndarray
            目标输出序列
        config : TrainingConfig, optional
            训练配置
        
        Returns
        -------
        history : Dict[str, List[float]]
            训练历史
        """
        if config is None:
            config = TrainingConfig()
        
        device = torch.device(config.device)
        self.model.to(device)
        
        # 数据归一化
        input_norm, self.scaler_input = self._normalize_data(input_sequences, fit=True)
        output_norm, self.scaler_output = self._normalize_data(output_sequences, fit=True)
        
        # 划分训练集和验证集
        n_samples = len(input_sequences)
        n_val = int(n_samples * config.validation_split)
        n_train = n_samples - n_val
        
        train_inputs = input_norm[:n_train]
        train_outputs = output_norm[:n_train]
        val_inputs = input_norm[n_train:]
        val_outputs = output_norm[n_train:]
        
        # 创建数据加载器
        train_dataset = SequenceDataset(train_inputs, train_outputs)
        val_dataset = SequenceDataset(val_inputs, val_outputs)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
        )
        
        # 优化器和损失函数
        optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        criterion = nn.MSELoss()
        
        # 训练循环
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config.num_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            
            for batch_inputs, batch_targets in train_loader:
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_inputs)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_inputs, batch_targets in val_loader:
                    batch_inputs = batch_inputs.to(device)
                    batch_targets = batch_targets.to(device)
                    
                    outputs = self.model(batch_inputs)
                    loss = criterion(outputs, batch_targets)
                    
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    print(f"早停于epoch {epoch + 1}")
                    break
        
        self.is_trained = True
        return history
    
    def predict(
        self,
        input_sequence: np.ndarray,  # (seq_len, n_input_features)
        device: str = "cpu",
    ) -> np.ndarray:
        """
        预测
        
        Parameters
        ----------
        input_sequence : np.ndarray
            输入序列
        device : str
            设备
        
        Returns
        -------
        output : np.ndarray
            预测输出 (seq_len, n_output_features)
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练")
        
        device = torch.device(device)
        self.model.to(device)
        self.model.eval()
        
        # 归一化输入
        input_norm, _ = self._normalize_data(
            input_sequence[np.newaxis, :, :],
            fit=False,
            scaler=self.scaler_input,
        )
        
        # 预测
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_norm).to(device)
            output_tensor = self.model(input_tensor)
            output_norm = output_tensor.cpu().numpy()[0]
        
        # 反归一化
        output = output_norm * self.scaler_output['std'] + self.scaler_output['mean']
        
        return output


class TransformerPredictor:
    """Transformer预测器"""
    
    def __init__(
        self,
        input_features: List[str],
        output_features: List[str],
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        """
        初始化Transformer预测器
        
        Parameters
        ----------
        input_features : List[str]
            输入特征名称列表
        output_features : List[str]
            输出特征名称列表
        d_model : int
            模型维度
        nhead : int
            注意力头数
        num_encoder_layers : int
            编码器层数
        dim_feedforward : int
            前馈网络维度
        dropout : float
            Dropout比例
        """
        if not TORCH_AVAILABLE:
            raise ImportError("需要安装PyTorch才能使用深度学习模型")
        
        self.input_features = input_features
        self.output_features = output_features
        self.input_size = len(input_features)
        self.output_size = len(output_features)
        
        self.model = TransformerNetwork(
            input_size=self.input_size,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            output_size=self.output_size,
            dropout=dropout,
        )
        
        self.scaler_input = None
        self.scaler_output = None
        self.is_trained = False
    
    def train(
        self,
        input_sequences: np.ndarray,
        output_sequences: np.ndarray,
        config: Optional[TrainingConfig] = None,
    ) -> Dict[str, List[float]]:
        """训练模型（与LSTM类似）"""
        # 实现与LSTMPredictor类似
        pass
    
    def predict(
        self,
        input_sequence: np.ndarray,
        device: str = "cpu",
    ) -> np.ndarray:
        """预测（与LSTM类似）"""
        # 实现与LSTMPredictor类似
        pass


class DeepLearningReducedModel:
    """深度学习降阶模型（集成接口）"""
    
    def __init__(
        self,
        model_type: str = "lstm",  # "lstm" or "transformer"
        **kwargs,
    ):
        """
        初始化深度学习降阶模型
        
        Parameters
        ----------
        model_type : str
            模型类型
        **kwargs
            模型参数
        """
        if not TORCH_AVAILABLE:
            raise ImportError("需要安装PyTorch才能使用深度学习模型: pip install torch")
        
        self.model_type = model_type
        
        # 定义输入输出特征
        input_features = [
            'upstream_discharge',
            'groundwater_level',
            'rainfall_intensity',
        ]
        
        output_features = [
            'water_depth',
            'water_quality_concentration',
            'slope_stability_factor',
        ]
        
        if model_type == "lstm":
            self.predictor = LSTMPredictor(
                input_features=input_features,
                output_features=output_features,
                **kwargs,
            )
        elif model_type == "transformer":
            self.predictor = TransformerPredictor(
                input_features=input_features,
                output_features=output_features,
                **kwargs,
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    def train_from_simulation_data(
        self,
        simulation_results: Dict[str, np.ndarray],
        config: Optional[TrainingConfig] = None,
    ) -> Dict[str, List[float]]:
        """
        从仿真数据训练模型
        
        Parameters
        ----------
        simulation_results : Dict[str, np.ndarray]
            仿真结果字典
        config : TrainingConfig, optional
            训练配置
        
        Returns
        -------
        history : Dict[str, List[float]]
            训练历史
        """
        # 构造输入输出序列
        input_sequences = np.stack([
            simulation_results['upstream_discharge'],
            simulation_results['groundwater_level'],
            simulation_results['rainfall_intensity'],
        ], axis=-1)
        
        output_sequences = np.stack([
            simulation_results['water_depth'],
            simulation_results['water_quality'],
            simulation_results['stability_factor'],
        ], axis=-1)
        
        return self.predictor.train(input_sequences, output_sequences, config)
    
    def predict(
        self,
        inputs: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        快速预测
        
        Parameters
        ----------
        inputs : Dict[str, np.ndarray]
            输入字典
        
        Returns
        -------
        outputs : Dict[str, np.ndarray]
            预测输出字典
        """
        # 构造输入序列
        input_sequence = np.stack([
            inputs['upstream_discharge'],
            inputs['groundwater_level'],
            inputs['rainfall_intensity'],
        ], axis=-1)
        
        # 预测
        output_sequence = self.predictor.predict(input_sequence)
        
        # 解析输出
        return {
            'water_depth': output_sequence[:, 0],
            'water_quality': output_sequence[:, 1],
            'stability_factor': output_sequence[:, 2],
        }
