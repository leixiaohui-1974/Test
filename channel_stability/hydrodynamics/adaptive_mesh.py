"""
自适应断面划分算法

根据水力学计算的稳定性和精度要求，动态调整计算断面的分布
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class MeshQualityMetrics:
    """网格质量指标"""
    max_depth_gradient: float  # 最大水深梯度
    max_velocity_gradient: float  # 最大流速梯度
    max_froude_number: float  # 最大Froude数
    min_cell_size: float  # 最小网格尺寸
    max_cell_size: float  # 最大网格尺寸
    refinement_needed: bool  # 是否需要加密
    coarsening_needed: bool  # 是否需要稀疏化


class AdaptiveMeshRefiner:
    """自适应网格加密算法"""
    
    def __init__(
        self,
        min_dx: float = 10.0,  # 最小网格间距 (m)
        max_dx: float = 1000.0,  # 最大网格间距 (m)
        target_dx: float = 100.0,  # 目标网格间距 (m)
        depth_gradient_threshold: float = 0.01,  # 水深梯度阈值
        velocity_gradient_threshold: float = 0.1,  # 流速梯度阈值 (m/s per m)
        froude_threshold: float = 0.8,  # Froude数阈值
    ):
        """
        初始化自适应网格加密器
        
        Parameters
        ----------
        min_dx : float
            最小网格间距
        max_dx : float
            最大网格间距
        target_dx : float
            目标网格间距
        depth_gradient_threshold : float
            水深梯度阈值，超过此值需要加密
        velocity_gradient_threshold : float
            流速梯度阈值
        froude_threshold : float
            Froude数阈值，接近临界流时需要加密
        """
        self.min_dx = min_dx
        self.max_dx = max_dx
        self.target_dx = target_dx
        self.depth_gradient_threshold = depth_gradient_threshold
        self.velocity_gradient_threshold = velocity_gradient_threshold
        self.froude_threshold = froude_threshold
    
    def evaluate_mesh_quality(
        self,
        stations: np.ndarray,
        depths: np.ndarray,
        discharges: np.ndarray,
        areas: np.ndarray,
        top_widths: np.ndarray,
    ) -> MeshQualityMetrics:
        """
        评估当前网格质量
        
        Parameters
        ----------
        stations : np.ndarray
            断面桩号
        depths : np.ndarray
            水深
        discharges : np.ndarray
            流量
        areas : np.ndarray
            过流面积
        top_widths : np.ndarray
            水面宽度
        
        Returns
        -------
        metrics : MeshQualityMetrics
            网格质量指标
        """
        n = len(stations)
        if n < 2:
            return MeshQualityMetrics(
                max_depth_gradient=0.0,
                max_velocity_gradient=0.0,
                max_froude_number=0.0,
                min_cell_size=0.0,
                max_cell_size=0.0,
                refinement_needed=False,
                coarsening_needed=False,
            )
        
        dx = np.diff(stations)
        
        # 计算水深梯度
        depth_gradients = np.abs(np.diff(depths) / dx)
        max_depth_gradient = np.max(depth_gradients) if len(depth_gradients) > 0 else 0.0
        
        # 计算流速和流速梯度
        velocities = np.where(areas > 1e-6, discharges / areas, 0.0)
        velocity_gradients = np.abs(np.diff(velocities) / dx)
        max_velocity_gradient = np.max(velocity_gradients) if len(velocity_gradients) > 0 else 0.0
        
        # 计算Froude数
        g = 9.81
        froude_numbers = np.where(
            (depths > 1e-3) & (top_widths > 1e-3),
            velocities / np.sqrt(g * areas / top_widths),
            0.0,
        )
        max_froude_number = np.max(froude_numbers)
        
        # 网格尺寸统计
        min_cell_size = np.min(dx)
        max_cell_size = np.max(dx)
        
        # 判断是否需要加密或稀疏化
        refinement_needed = (
            max_depth_gradient > self.depth_gradient_threshold or
            max_velocity_gradient > self.velocity_gradient_threshold or
            max_froude_number > self.froude_threshold
        )
        
        coarsening_needed = (
            max_depth_gradient < 0.5 * self.depth_gradient_threshold and
            max_velocity_gradient < 0.5 * self.velocity_gradient_threshold and
            max_cell_size > 2 * self.target_dx
        )
        
        return MeshQualityMetrics(
            max_depth_gradient=max_depth_gradient,
            max_velocity_gradient=max_velocity_gradient,
            max_froude_number=max_froude_number,
            min_cell_size=min_cell_size,
            max_cell_size=max_cell_size,
            refinement_needed=refinement_needed,
            coarsening_needed=coarsening_needed,
        )
    
    def compute_refinement_indicators(
        self,
        stations: np.ndarray,
        depths: np.ndarray,
        discharges: np.ndarray,
        areas: np.ndarray,
        top_widths: np.ndarray,
    ) -> np.ndarray:
        """
        计算各单元的加密指示器
        
        Returns
        -------
        indicators : np.ndarray
            加密指示器，值越大表示越需要加密
        """
        n = len(stations)
        if n < 2:
            return np.array([])
        
        dx = np.diff(stations)
        
        # 水深梯度贡献
        depth_gradients = np.abs(np.diff(depths) / dx)
        depth_indicator = depth_gradients / (self.depth_gradient_threshold + 1e-10)
        
        # 流速梯度贡献
        velocities = np.where(areas > 1e-6, discharges / areas, 0.0)
        velocity_gradients = np.abs(np.diff(velocities) / dx)
        velocity_indicator = velocity_gradients / (self.velocity_gradient_threshold + 1e-10)
        
        # Froude数贡献（在单元中心计算）
        g = 9.81
        froude_numbers = np.where(
            (depths > 1e-3) & (top_widths > 1e-3),
            velocities / np.sqrt(g * areas / top_widths),
            0.0,
        )
        # 将Froude数投影到单元上（取相邻节点的平均）
        froude_cells = 0.5 * (froude_numbers[:-1] + froude_numbers[1:])
        froude_indicator = np.maximum(
            froude_cells / self.froude_threshold,
            (1.0 - froude_cells) / (1.0 - self.froude_threshold),
        )
        
        # 综合指示器（取最大值）
        indicators = np.maximum(
            np.maximum(depth_indicator, velocity_indicator),
            froude_indicator,
        )
        
        return indicators
    
    def refine_mesh(
        self,
        stations: np.ndarray,
        indicators: np.ndarray,
        refinement_threshold: float = 1.0,
    ) -> np.ndarray:
        """
        根据加密指示器细化网格
        
        Parameters
        ----------
        stations : np.ndarray
            当前断面桩号
        indicators : np.ndarray
            加密指示器
        refinement_threshold : float
            加密阈值
        
        Returns
        -------
        new_stations : np.ndarray
            新的断面桩号
        """
        if len(stations) < 2:
            return stations
        
        new_stations = [stations[0]]
        
        for i in range(len(indicators)):
            current_station = stations[i]
            next_station = stations[i + 1]
            dx = next_station - current_station
            
            # 判断是否需要加密
            if indicators[i] > refinement_threshold and dx > self.min_dx * 2:
                # 在中点插入新断面
                mid_station = 0.5 * (current_station + next_station)
                new_stations.append(mid_station)
            
            new_stations.append(next_station)
        
        return np.array(new_stations)
    
    def coarsen_mesh(
        self,
        stations: np.ndarray,
        indicators: np.ndarray,
        coarsening_threshold: float = 0.3,
    ) -> np.ndarray:
        """
        根据指示器粗化网格（移除不必要的断面）
        
        Parameters
        ----------
        stations : np.ndarray
            当前断面桩号
        indicators : np.ndarray
            加密指示器
        coarsening_threshold : float
            粗化阈值
        
        Returns
        -------
        new_stations : np.ndarray
            新的断面桩号
        """
        if len(stations) < 3:
            return stations
        
        # 始终保留第一个和最后一个断面
        keep_indices = [0]
        
        for i in range(1, len(stations) - 1):
            # 检查前后单元的指示器
            prev_indicator = indicators[i - 1] if i - 1 < len(indicators) else 1.0
            next_indicator = indicators[i] if i < len(indicators) else 1.0
            
            # 如果两侧的指示器都很小，考虑移除该断面
            if prev_indicator < coarsening_threshold and next_indicator < coarsening_threshold:
                # 检查移除后的网格间距是否超过最大值
                prev_station = stations[keep_indices[-1]]
                next_station = stations[i + 1]
                if next_station - prev_station <= self.max_dx:
                    # 可以移除
                    continue
            
            keep_indices.append(i)
        
        keep_indices.append(len(stations) - 1)
        
        return stations[keep_indices]
    
    def adapt_mesh(
        self,
        stations: np.ndarray,
        depths: np.ndarray,
        discharges: np.ndarray,
        areas: np.ndarray,
        top_widths: np.ndarray,
        max_iterations: int = 3,
    ) -> Tuple[np.ndarray, MeshQualityMetrics]:
        """
        自适应调整网格
        
        Parameters
        ----------
        stations : np.ndarray
            当前断面桩号
        depths : np.ndarray
            水深
        discharges : np.ndarray
            流量
        areas : np.ndarray
            过流面积
        top_widths : np.ndarray
            水面宽度
        max_iterations : int
            最大迭代次数
        
        Returns
        -------
        new_stations : np.ndarray
            调整后的断面桩号
        metrics : MeshQualityMetrics
            网格质量指标
        """
        current_stations = stations.copy()
        
        for iteration in range(max_iterations):
            # 评估网格质量
            metrics = self.evaluate_mesh_quality(
                current_stations, depths, discharges, areas, top_widths
            )
            
            # 如果不需要调整，直接返回
            if not metrics.refinement_needed and not metrics.coarsening_needed:
                break
            
            # 计算加密指示器
            indicators = self.compute_refinement_indicators(
                current_stations, depths, discharges, areas, top_widths
            )
            
            # 先加密，后粗化
            if metrics.refinement_needed:
                current_stations = self.refine_mesh(
                    current_stations, indicators, refinement_threshold=1.0
                )
            
            if metrics.coarsening_needed and iteration < max_iterations - 1:
                current_stations = self.coarsen_mesh(
                    current_stations, indicators, coarsening_threshold=0.3
                )
        
        # 最终评估
        metrics = self.evaluate_mesh_quality(
            current_stations, depths, discharges, areas, top_widths
        )
        
        return current_stations, metrics
    
    def create_initial_mesh(
        self,
        channel_length: float,
        base_dx: Optional[float] = None,
        refinement_regions: Optional[List[Tuple[float, float, float]]] = None,
    ) -> np.ndarray:
        """
        创建初始网格
        
        Parameters
        ----------
        channel_length : float
            渠道长度
        base_dx : float, optional
            基础网格间距，默认使用target_dx
        refinement_regions : List[Tuple[float, float, float]], optional
            需要加密的区域列表，每个元素为(start, end, dx)
        
        Returns
        -------
        stations : np.ndarray
            断面桩号
        """
        if base_dx is None:
            base_dx = self.target_dx
        
        # 创建基础均匀网格
        num_base = int(channel_length / base_dx) + 1
        base_stations = np.linspace(0, channel_length, num_base)
        
        # 如果没有指定加密区域，直接返回基础网格
        if refinement_regions is None or not refinement_regions:
            return base_stations
        
        # 在加密区域插入额外的断面
        all_stations = list(base_stations)
        
        for start, end, dx_refined in refinement_regions:
            num_refined = int((end - start) / dx_refined) + 1
            refined_stations = np.linspace(start, end, num_refined)
            all_stations.extend(refined_stations)
        
        # 去重并排序
        all_stations = np.unique(all_stations)
        
        return all_stations
