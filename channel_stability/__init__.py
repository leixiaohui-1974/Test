"""
明渠边坡稳定性监测预测系统
==========================

本系统提供明渠水力学、水质和边坡稳定性的集成仿真能力。

主要模块：
- hydrodynamics: 水动力学模块（Preissmann隐式格式+自适应断面划分）
- water_quality: 水质模拟模块（对流-扩散-反应方程）
- slope_stability: 边坡稳定性计算模块
- reduced_order: 降阶模型（线性化+深度学习）
- integrated_simulation: 集成仿真系统
"""

__version__ = "1.0.0"
__author__ = "Water Resources Engineering Team"

from .core.channel_system import ChannelSystem
from .core.monitoring_network import MonitoringNetwork
from .integrated_simulation import IntegratedSimulator

__all__ = [
    "ChannelSystem",
    "MonitoringNetwork",
    "IntegratedSimulator",
]
