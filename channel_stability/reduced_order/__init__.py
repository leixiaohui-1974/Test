"""
降阶模型模块

提供线性化降阶和深度学习降阶两种方法
用于快速预测和实时应用
"""

from .linear_reduced_model import (
    LinearizedHydrodynamics,
    LinearizedWaterQuality,
    LinearizedSlopeStability,
)
from .deep_learning_model import (
    DeepLearningReducedModel,
    LSTMPredictor,
    TransformerPredictor,
)

__all__ = [
    "LinearizedHydrodynamics",
    "LinearizedWaterQuality",
    "LinearizedSlopeStability",
    "DeepLearningReducedModel",
    "LSTMPredictor",
    "TransformerPredictor",
]
