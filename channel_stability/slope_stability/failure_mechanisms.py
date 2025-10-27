"""
边坡失稳机理分析

定义衬砌板的多种失稳机制
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import numpy as np

from ..core.channel_system import SoilProperties, LiningProperties


class FailureMechanism(Enum):
    """失稳机制"""
    SLIDING = "sliding"  # 滑动失稳
    OVERTURNING = "overturning"  # 倾覆失稳
    UPLIFT = "uplift"  # 浮托失稳
    SEEPAGE = "seepage"  # 渗透失稳


@dataclass
class StabilityFactors:
    """稳定性系数"""
    sliding_factor: float  # 抗滑稳定系数
    overturning_factor: float  # 抗倾覆稳定系数
    uplift_factor: float  # 抗浮稳定系数
    seepage_factor: float  # 渗透稳定系数
    
    comprehensive_factor: float  # 综合稳定系数
    is_stable: bool  # 是否稳定
    
    def to_dict(self):
        return {
            'sliding_factor': self.sliding_factor,
            'overturning_factor': self.overturning_factor,
            'uplift_factor': self.uplift_factor,
            'seepage_factor': self.seepage_factor,
            'comprehensive_factor': self.comprehensive_factor,
            'is_stable': self.is_stable,
        }


class LiningStabilityAnalysis:
    """衬砌板稳定性分析"""
    
    # 安全系数阈值
    SLIDING_THRESHOLD = 1.3  # 抗滑安全系数
    OVERTURNING_THRESHOLD = 1.5  # 抗倾覆安全系数
    UPLIFT_THRESHOLD = 1.2  # 抗浮安全系数
    SEEPAGE_THRESHOLD = 1.5  # 渗透稳定系数
    
    @staticmethod
    def compute_sliding_stability(
        slope_angle: float,  # 边坡角度 (度)
        slope_height: float,  # 边坡高度 (m)
        soil_props: SoilProperties,
        lining_props: LiningProperties,
        groundwater_depth: float,  # 地下水埋深 (m，从边坡顶部)
        channel_water_level: float,  # 渠道水位 (m，相对于渠底)
        rainfall_intensity: float = 0.0,  # 降雨强度 (mm/h)
    ) -> float:
        """
        计算抗滑稳定系数
        
        基于摩擦圆法或条分法
        
        Returns
        -------
        factor : float
            抗滑稳定系数
        """
        # 边坡角度转换为弧度
        alpha = np.radians(slope_angle)
        
        # 衬砌板重量 (kN/m)
        lining_weight = lining_props.thickness * lining_props.density * 9.81 / 1000.0
        
        # 土体重量（饱和区和非饱和区分开计算）
        if groundwater_depth <= 0:
            # 全部饱和
            saturated_height = slope_height
            unsaturated_height = 0.0
        elif groundwater_depth >= slope_height:
            # 完全非饱和
            saturated_height = 0.0
            unsaturated_height = slope_height
        else:
            saturated_height = slope_height - groundwater_depth
            unsaturated_height = groundwater_depth
        
        # 土体重量 (kN/m)
        soil_weight = (
            unsaturated_height * soil_props.unit_weight +
            saturated_height * soil_props.saturated_unit_weight
        ) * np.cos(alpha)
        
        # 总正压力
        total_weight = lining_weight + soil_weight
        
        # 考虑渗透压力（雨水入渗）
        if rainfall_intensity > 0:
            # 简化模型：降雨增加孔隙水压
            rainfall_factor = 1.0 + 0.01 * rainfall_intensity  # 经验公式
            pore_pressure_factor = rainfall_factor
        else:
            pore_pressure_factor = 1.0
        
        # 地下水产生的孔隙水压力（简化）
        if groundwater_depth < slope_height:
            water_pressure = 9.81 * saturated_height * pore_pressure_factor
        else:
            water_pressure = 0.0
        
        # 有效正压力
        effective_normal_force = total_weight - water_pressure * np.sin(alpha)
        
        # 抗滑力
        cohesion_force = soil_props.cohesion  # kPa -> kN/m²
        friction_force = effective_normal_force * np.tan(np.radians(soil_props.friction_angle))
        friction_force += lining_weight * lining_props.friction_coeff
        
        resisting_force = cohesion_force * slope_height / np.cos(alpha) + friction_force
        
        # 下滑力
        driving_force = total_weight * np.sin(alpha)
        
        # 稳定系数
        if driving_force > 1e-6:
            factor = resisting_force / driving_force
        else:
            factor = 10.0  # 下滑力很小，认为非常稳定
        
        return min(max(factor, 0.0), 10.0)  # 限制在0-10范围
    
    @staticmethod
    def compute_overturning_stability(
        slope_angle: float,
        slope_height: float,
        soil_props: SoilProperties,
        lining_props: LiningProperties,
        groundwater_depth: float,
        channel_water_level: float,
    ) -> float:
        """
        计算抗倾覆稳定系数
        
        Returns
        -------
        factor : float
            抗倾覆稳定系数
        """
        alpha = np.radians(slope_angle)
        
        # 衬砌板重量和重心
        lining_weight = lining_props.thickness * lining_props.density * 9.81 / 1000.0
        lining_arm = 0.5 * slope_height * np.cos(alpha)  # 到倾覆点的力臂
        
        # 土体重量和重心
        if groundwater_depth <= 0:
            saturated_height = slope_height
            unsaturated_height = 0.0
        elif groundwater_depth >= slope_height:
            saturated_height = 0.0
            unsaturated_height = slope_height
        else:
            saturated_height = slope_height - groundwater_depth
            unsaturated_height = groundwater_depth
        
        soil_weight = (
            unsaturated_height * soil_props.unit_weight +
            saturated_height * soil_props.saturated_unit_weight
        )
        soil_arm = 0.5 * slope_height * np.cos(alpha)
        
        # 抗倾覆力矩
        resisting_moment = lining_weight * lining_arm + soil_weight * soil_arm
        
        # 倾覆力矩（地下水压力）
        if groundwater_depth < slope_height:
            water_height = slope_height - groundwater_depth
            water_force = 0.5 * 9.81 * water_height ** 2
            water_arm = water_height / 3.0
            overturning_moment = water_force * water_arm
        else:
            # 无地下水，无倾覆力矩
            return 10.0  # 非常稳定
        
        # 稳定系数
        if overturning_moment > 1e-6:
            factor = resisting_moment / overturning_moment
        else:
            factor = 10.0
        
        return min(max(factor, 0.0), 10.0)  # 限制在0-10范围
    
    @staticmethod
    def compute_uplift_stability(
        slope_angle: float,
        slope_height: float,
        soil_props: SoilProperties,
        lining_props: LiningProperties,
        groundwater_depth: float,
        channel_water_level: float,
    ) -> float:
        """
        计算抗浮稳定系数
        
        Returns
        -------
        factor : float
            抗浮稳定系数
        """
        # 衬砌板自重
        lining_weight = lining_props.thickness * lining_props.density * 9.81 / 1000.0
        
        # 土体自重
        if groundwater_depth >= slope_height:
            soil_weight = slope_height * soil_props.unit_weight
            uplift_force = 1e-6  # 无浮托力
        else:
            saturated_height = slope_height - groundwater_depth
            unsaturated_height = groundwater_depth
            
            soil_weight = (
                unsaturated_height * soil_props.unit_weight +
                saturated_height * soil_props.saturated_unit_weight
            )
            
            # 浮托力（地下水位以下）
            uplift_force = saturated_height * 9.81
        
        # 总重量
        total_weight = lining_weight + soil_weight
        
        # 稳定系数
        if uplift_force > 1e-6:
            factor = total_weight / uplift_force
        else:
            factor = 10.0  # 无浮托力，非常稳定
        
        return min(max(factor, 0.0), 10.0)  # 限制在0-10范围
    
    @staticmethod
    def compute_seepage_stability(
        slope_angle: float,
        slope_height: float,
        soil_props: SoilProperties,
        groundwater_depth: float,
        channel_water_level: float,
    ) -> float:
        """
        计算渗透稳定系数
        
        基于渗透力与土体抗剪强度的比值
        
        Returns
        -------
        factor : float
            渗透稳定系数
        """
        if groundwater_depth >= slope_height:
            # 无渗透
            return 10.0
        
        # 水力梯度（简化计算）
        water_level_diff = max(channel_water_level, 0.0)
        seepage_path = slope_height * np.cos(np.radians(slope_angle))
        
        if seepage_path > 0:
            hydraulic_gradient = water_level_diff / seepage_path
        else:
            hydraulic_gradient = 0.0
        
        # 临界水力梯度
        critical_gradient = (soil_props.saturated_unit_weight - 9.81) / 9.81
        
        # 稳定系数
        if hydraulic_gradient > 1e-6:
            factor = critical_gradient / hydraulic_gradient
        else:
            factor = 10.0  # 无渗透，非常稳定
        
        return min(max(factor, 0.0), 10.0)  # 限制在0-10范围
    
    @staticmethod
    def compute_comprehensive_stability(
        slope_angle: float,
        slope_height: float,
        soil_props: SoilProperties,
        lining_props: LiningProperties,
        groundwater_depth: float,
        channel_water_level: float,
        rainfall_intensity: float = 0.0,
    ) -> StabilityFactors:
        """
        计算综合稳定性
        
        Returns
        -------
        factors : StabilityFactors
            稳定性系数
        """
        # 计算各失稳模式的稳定系数
        sliding = LiningStabilityAnalysis.compute_sliding_stability(
            slope_angle, slope_height, soil_props, lining_props,
            groundwater_depth, channel_water_level, rainfall_intensity
        )
        
        overturning = LiningStabilityAnalysis.compute_overturning_stability(
            slope_angle, slope_height, soil_props, lining_props,
            groundwater_depth, channel_water_level
        )
        
        uplift = LiningStabilityAnalysis.compute_uplift_stability(
            slope_angle, slope_height, soil_props, lining_props,
            groundwater_depth, channel_water_level
        )
        
        seepage = LiningStabilityAnalysis.compute_seepage_stability(
            slope_angle, slope_height, soil_props,
            groundwater_depth, channel_water_level
        )
        
        # 综合稳定系数（加权平均或最小值法）
        # 这里采用最小值法（最不利情况）
        comprehensive = min(sliding, overturning, uplift, seepage)
        
        # 判断是否稳定（所有系数都满足要求）
        is_stable = (
            sliding >= LiningStabilityAnalysis.SLIDING_THRESHOLD and
            overturning >= LiningStabilityAnalysis.OVERTURNING_THRESHOLD and
            uplift >= LiningStabilityAnalysis.UPLIFT_THRESHOLD and
            seepage >= LiningStabilityAnalysis.SEEPAGE_THRESHOLD
        )
        
        return StabilityFactors(
            sliding_factor=sliding,
            overturning_factor=overturning,
            uplift_factor=uplift,
            seepage_factor=seepage,
            comprehensive_factor=comprehensive,
            is_stable=is_stable,
        )
