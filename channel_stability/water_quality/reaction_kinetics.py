"""
水质反应动力学

定义常见水质指标的反应动力学过程
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class WaterQualityParameters:
    """水质参数"""
    
    # 温度
    temperature: float = 20.0  # 水温 (°C)
    
    # DO (溶解氧)
    do_saturation: float = 9.0  # 饱和溶解氧 (mg/L)
    k_reaeration: float = 0.5  # 复氧系数 (1/day)
    
    # BOD (生化需氧量)
    k_bod_decay: float = 0.2  # BOD降解系数 (1/day)
    
    # COD (化学需氧量)
    k_cod_decay: float = 0.15  # COD降解系数 (1/day)
    
    # 氨氮 (NH3-N)
    k_nitrification: float = 0.3  # 硝化系数 (1/day)
    
    # 总氮 (TN)
    k_denitrification: float = 0.1  # 反硝化系数 (1/day)
    k_nitrogen_settling: float = 0.05  # 氮沉降系数 (1/day)
    
    # 总磷 (TP)
    k_phosphorus_settling: float = 0.08  # 磷沉降系数 (1/day)
    
    # 悬浮物 (SS)
    k_ss_settling: float = 0.2  # 悬浮物沉降系数 (1/day)
    
    def temperature_correction(self, k20: float, theta: float = 1.024) -> float:
        """温度修正"""
        return k20 * (theta ** (self.temperature - 20))


class ReactionKinetics:
    """水质反应动力学"""
    
    @staticmethod
    def compute_do_source(
        do: float,
        bod: float,
        params: WaterQualityParameters,
    ) -> float:
        """
        计算溶解氧(DO)的源项
        
        Parameters
        ----------
        do : float
            当前DO浓度 (mg/L)
        bod : float
            当前BOD浓度 (mg/L)
        params : WaterQualityParameters
            水质参数
        
        Returns
        -------
        source : float
            DO源项 (mg/L/day)
        """
        # 复氧项（大气复氧）
        k_reaeration = params.temperature_correction(params.k_reaeration)
        reaeration = k_reaeration * (params.do_saturation - do)
        
        # 耗氧项（BOD降解消耗）
        k_bod = params.temperature_correction(params.k_bod_decay)
        bod_consumption = -k_bod * bod
        
        return reaeration + bod_consumption
    
    @staticmethod
    def compute_bod_source(
        bod: float,
        params: WaterQualityParameters,
    ) -> float:
        """
        计算生化需氧量(BOD)的源项
        
        Parameters
        ----------
        bod : float
            当前BOD浓度 (mg/L)
        params : WaterQualityParameters
            水质参数
        
        Returns
        -------
        source : float
            BOD源项 (mg/L/day)
        """
        k_bod = params.temperature_correction(params.k_bod_decay)
        return -k_bod * bod
    
    @staticmethod
    def compute_cod_source(
        cod: float,
        params: WaterQualityParameters,
    ) -> float:
        """
        计算化学需氧量(COD)的源项
        
        Parameters
        ----------
        cod : float
            当前COD浓度 (mg/L)
        params : WaterQualityParameters
            水质参数
        
        Returns
        -------
        source : float
            COD源项 (mg/L/day)
        """
        k_cod = params.temperature_correction(params.k_cod_decay)
        return -k_cod * cod
    
    @staticmethod
    def compute_nh3n_source(
        nh3n: float,
        do: float,
        params: WaterQualityParameters,
    ) -> float:
        """
        计算氨氮(NH3-N)的源项
        
        Parameters
        ----------
        nh3n : float
            当前NH3-N浓度 (mg/L)
        do : float
            溶解氧浓度 (mg/L)
        params : WaterQualityParameters
            水质参数
        
        Returns
        -------
        source : float
            NH3-N源项 (mg/L/day)
        """
        # 硝化作用（需要氧气）
        k_nitrification = params.temperature_correction(params.k_nitrification)
        
        # 氧气限制因子
        do_factor = do / (do + 0.5) if do > 0 else 0.0
        
        nitrification = -k_nitrification * nh3n * do_factor
        
        return nitrification
    
    @staticmethod
    def compute_tn_source(
        tn: float,
        do: float,
        params: WaterQualityParameters,
    ) -> float:
        """
        计算总氮(TN)的源项
        
        Parameters
        ----------
        tn : float
            当前TN浓度 (mg/L)
        do : float
            溶解氧浓度 (mg/L)
        params : WaterQualityParameters
            水质参数
        
        Returns
        -------
        source : float
            TN源项 (mg/L/day)
        """
        # 反硝化作用（缺氧条件）
        k_denitrification = params.temperature_correction(params.k_denitrification)
        do_factor = 1.0 / (do + 0.5) if do < 2.0 else 0.0
        denitrification = -k_denitrification * tn * do_factor
        
        # 沉降作用
        k_settling = params.temperature_correction(params.k_nitrogen_settling)
        settling = -k_settling * tn
        
        return denitrification + settling
    
    @staticmethod
    def compute_tp_source(
        tp: float,
        params: WaterQualityParameters,
    ) -> float:
        """
        计算总磷(TP)的源项
        
        Parameters
        ----------
        tp : float
            当前TP浓度 (mg/L)
        params : WaterQualityParameters
            水质参数
        
        Returns
        -------
        source : float
            TP源项 (mg/L/day)
        """
        # 沉降作用
        k_settling = params.temperature_correction(params.k_phosphorus_settling)
        return -k_settling * tp
    
    @staticmethod
    def compute_ss_source(
        ss: float,
        velocity: float,
        params: WaterQualityParameters,
    ) -> float:
        """
        计算悬浮物(SS)的源项
        
        Parameters
        ----------
        ss : float
            当前SS浓度 (mg/L)
        velocity : float
            流速 (m/s)
        params : WaterQualityParameters
            水质参数
        
        Returns
        -------
        source : float
            SS源项 (mg/L/day)
        """
        # 沉降作用（流速影响沉降效率）
        k_settling = params.temperature_correction(params.k_ss_settling)
        
        # 流速修正因子（流速越大，沉降越慢）
        velocity_factor = 1.0 / (1.0 + velocity)
        
        settling = -k_settling * ss * velocity_factor
        
        return settling
    
    @staticmethod
    def compute_all_sources(
        concentrations: Dict[str, float],
        velocity: float,
        params: WaterQualityParameters,
    ) -> Dict[str, float]:
        """
        计算所有水质指标的源项
        
        Parameters
        ----------
        concentrations : Dict[str, float]
            当前浓度字典
        velocity : float
            流速 (m/s)
        params : WaterQualityParameters
            水质参数
        
        Returns
        -------
        sources : Dict[str, float]
            源项字典 (mg/L/day)
        """
        do = concentrations.get('DO', 9.0)
        bod = concentrations.get('BOD', 0.0)
        cod = concentrations.get('COD', 0.0)
        nh3n = concentrations.get('NH3N', 0.0)
        tn = concentrations.get('TN', 0.0)
        tp = concentrations.get('TP', 0.0)
        ss = concentrations.get('SS', 0.0)
        
        sources = {}
        
        if 'DO' in concentrations:
            sources['DO'] = ReactionKinetics.compute_do_source(do, bod, params)
        
        if 'BOD' in concentrations:
            sources['BOD'] = ReactionKinetics.compute_bod_source(bod, params)
        
        if 'COD' in concentrations:
            sources['COD'] = ReactionKinetics.compute_cod_source(cod, params)
        
        if 'NH3N' in concentrations:
            sources['NH3N'] = ReactionKinetics.compute_nh3n_source(nh3n, do, params)
        
        if 'TN' in concentrations:
            sources['TN'] = ReactionKinetics.compute_tn_source(tn, do, params)
        
        if 'TP' in concentrations:
            sources['TP'] = ReactionKinetics.compute_tp_source(tp, params)
        
        if 'SS' in concentrations:
            sources['SS'] = ReactionKinetics.compute_ss_source(ss, velocity, params)
        
        return sources
