"""
水动力学模块

基于Preissmann隐式格式的一维非恒定流模拟
包含自适应断面划分功能
"""

from .preissmann_solver import PreissmannHydrodynamicSolver
from .adaptive_mesh import AdaptiveMeshRefiner
from .boundary_conditions import BoundaryConditions

__all__ = [
    "PreissmannHydrodynamicSolver",
    "AdaptiveMeshRefiner",
    "BoundaryConditions",
]
