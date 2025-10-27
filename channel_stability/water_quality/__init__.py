"""
水质模拟模块

基于对流-扩散-反应方程的一维水质数值模拟
"""

from .advection_diffusion_solver import WaterQualitySolver
from .reaction_kinetics import ReactionKinetics, WaterQualityParameters

__all__ = [
    "WaterQualitySolver",
    "ReactionKinetics",
    "WaterQualityParameters",
]
