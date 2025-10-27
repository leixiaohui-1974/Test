"""
明渠边坡稳定性监测预测系统示例

演示完整的仿真流程
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

from channel_stability.core.channel_system import (
    ChannelSystem,
    ChannelSection,
    SoilProperties,
    LiningProperties,
)
from channel_stability.core.monitoring_network import (
    MonitoringNetwork,
    MonitoringStation,
    StationType,
)
from channel_stability.hydrodynamics.boundary_conditions import BoundaryConditions
from channel_stability.water_quality.reaction_kinetics import WaterQualityParameters
from channel_stability.integrated_simulation import (
    IntegratedSimulator,
    SimulationConfig,
)


def create_example_channel() -> ChannelSystem:
    """创建示例明渠系统"""
    
    # 定义土壤特性
    soil = SoilProperties(
        cohesion=15.0,  # kPa
        friction_angle=30.0,  # 度
        unit_weight=18.0,  # kN/m³
        saturated_unit_weight=20.0,  # kN/m³
        permeability=1e-5,  # m/s
        porosity=0.35,
    )
    
    # 定义衬砌板特性
    lining = LiningProperties(
        thickness=0.1,  # m
        density=2400.0,  # kg/m³
        elastic_modulus=30.0,  # GPa
        friction_coeff=0.4,
    )
    
    # 创建明渠系统
    channel = ChannelSystem(
        name="示例明渠",
        total_length=20000.0,  # 20 km
        sections=[],
        description="高地下水位段明渠，20km长",
    )
    
    # 创建基准断面
    base_section = ChannelSection(
        station=0.0,
        bed_elevation=100.0,
        bottom_width=5.0,  # m
        side_slope=2.0,  # 1:2边坡
        max_depth=4.0,  # m
        manning_n=0.025,
        slope_height=3.0,  # m
        slope_angle=26.57,  # 度，对应1:2边坡
        soil_properties=soil,
        lining_properties=lining,
    )
    
    # 创建均匀分布的断面（每500m一个，共41个）
    channel.create_uniform_sections(
        num_sections=41,
        base_section=base_section,
        bed_slope=0.0001,  # 万分之一坡度
    )
    
    return channel


def create_monitoring_network(channel: ChannelSystem) -> MonitoringNetwork:
    """创建监测网络"""
    
    network = MonitoringNetwork(
        network_name="明渠监测网络",
        groundwater_spacing=500.0,
    )
    
    # 创建地下水观测井网络（每500m一个）
    network.create_uniform_groundwater_network(
        channel_length=channel.total_length,
        spacing=500.0,
    )
    
    # 添加渠道两端的雨量计和水位计
    network.add_boundary_stations(
        upstream_station=0.0,
        downstream_station=channel.total_length,
    )
    
    # 模拟观测数据
    simulate_monitoring_data(network)
    
    return network


def simulate_monitoring_data(network: MonitoringNetwork) -> None:
    """模拟监测数据"""
    
    # 模拟24小时的观测数据
    times = np.linspace(0, 86400, 100)  # 100个观测点
    
    # 地下水位（随时间和空间变化）
    for station in network.groundwater_stations:
        # 基准地下水位
        base_gw_level = 98.0  # m（接近地表）
        
        # 时间变化（降雨影响）
        time_variation = 0.5 * np.sin(2 * np.pi * times / 86400)
        
        # 空间变化
        spatial_factor = station.station / network.stations[-1].station
        
        for t, time_var in zip(times, time_variation):
            gw_level = base_gw_level + time_var + 0.2 * spatial_factor
            station.add_observation(
                time=t,
                groundwater_level=gw_level,
            )
    
    # 降雨数据（上下游站点）
    rainfall_pattern = np.zeros_like(times)
    # 模拟一次降雨事件（10-14小时）
    rain_start = int(len(times) * 10 / 24)
    rain_end = int(len(times) * 14 / 24)
    rainfall_pattern[rain_start:rain_end] = 10.0  # mm/h
    
    for station in network.rainfall_stations:
        for t, rainfall in zip(times, rainfall_pattern):
            station.add_observation(
                time=t,
                rainfall=rainfall,
            )


def create_boundary_conditions() -> BoundaryConditions:
    """创建边界条件"""
    
    # 模拟上游流量过程（包含洪峰）
    def upstream_discharge(t):
        # 基流 + 洪峰
        base_flow = 10.0  # m³/s
        peak_time = 43200.0  # 12小时
        peak_magnitude = 20.0  # m³/s
        
        # 高斯型洪峰
        flood_wave = peak_magnitude * np.exp(-((t - peak_time) / 7200.0) ** 2)
        
        return base_flow + flood_wave
    
    # 下游水位（假设恒定）
    downstream_h = 2.5  # m
    
    bc = BoundaryConditions(
        upstream_type="discharge",
        downstream_type="stage",
        upstream_discharge_func=upstream_discharge,
        downstream_constant_h=downstream_h,
    )
    
    return bc


def run_example_simulation():
    """运行示例仿真"""
    
    print("=" * 60)
    print("明渠边坡稳定性监测预测系统 - 案例演示")
    print("=" * 60)
    print()
    
    # 1. 创建明渠系统
    print("步骤 1: 创建明渠系统...")
    channel = create_example_channel()
    print(f"  明渠长度: {channel.total_length/1000:.1f} km")
    print(f"  断面数量: {channel.num_sections}")
    print()
    
    # 2. 创建监测网络
    print("步骤 2: 创建监测网络...")
    monitoring_network = create_monitoring_network(channel)
    print(f"  监测站总数: {monitoring_network.num_stations}")
    print(f"  地下水观测井: {len(monitoring_network.groundwater_stations)}")
    print(f"  雨量计: {len(monitoring_network.rainfall_stations)}")
    print()
    
    # 3. 定义边界条件
    print("步骤 3: 定义边界条件...")
    boundary_conditions = create_boundary_conditions()
    print("  上游: 流量过程（基流+洪峰）")
    print("  下游: 恒定水位")
    print()
    
    # 4. 设置水质参数
    print("步骤 4: 设置水质参数...")
    wq_params = WaterQualityParameters(temperature=20.0)
    print(f"  水温: {wq_params.temperature}°C")
    print()
    
    # 5. 创建综合仿真器
    print("步骤 5: 初始化综合仿真器...")
    simulator = IntegratedSimulator(
        channel=channel,
        monitoring_network=monitoring_network,
        boundary_conditions=boundary_conditions,
        water_quality_params=wq_params,
    )
    print("  仿真器已初始化")
    print()
    
    # 6. 配置仿真参数
    print("步骤 6: 配置仿真参数...")
    config = SimulationConfig(
        total_time=86400.0,  # 24小时
        dt=60.0,  # 1分钟
        save_interval=10,  # 每10步保存一次
        enable_hydrodynamics=True,
        enable_water_quality=True,
        enable_slope_stability=True,
        simulated_indicators=['DO', 'BOD', 'NH3N', 'TN', 'TP'],
        stability_time_interval=50,
    )
    print(f"  仿真时长: {config.total_time/3600:.1f} 小时")
    print(f"  时间步长: {config.dt} 秒")
    print()
    
    # 7. 运行仿真
    print("步骤 7: 运行综合仿真...")
    print("-" * 60)
    results = simulator.run_simulation(config)
    print("-" * 60)
    print()
    
    # 8. 结果摘要
    print("步骤 8: 结果摘要")
    print("=" * 60)
    summary = results.summary()
    
    if 'hydrodynamics' in summary:
        print("水动力学结果:")
        print(f"  时间步数: {summary['hydrodynamics']['num_times']}")
        print(f"  最大水深: {summary['hydrodynamics']['max_depth']:.2f} m")
        print(f"  最大流速: {summary['hydrodynamics']['max_velocity']:.2f} m/s")
        print(f"  最大Froude数: {summary['hydrodynamics']['max_froude']:.3f}")
        print()
    
    if 'water_quality' in summary:
        print("水质模拟结果:")
        print(f"  模拟指标: {', '.join(summary['water_quality']['indicators'])}")
        for indicator in summary['water_quality']['indicators']:
            max_val = summary['water_quality'][f'{indicator}_max']
            min_val = summary['water_quality'][f'{indicator}_min']
            print(f"  {indicator}: {min_val:.2f} - {max_val:.2f} mg/L")
        print()
    
    if 'slope_stability' in summary:
        print("边坡稳定性结果:")
        print(f"  最小稳定系数: {summary['slope_stability']['min_stability_factor']:.2f}")
        print(f"  最大不稳定断面数: {summary['slope_stability']['max_unstable_sections']}")
        print(f"  综合稳定性指标: {summary['slope_stability']['overall_stability_index']:.2f}")
        print()
    
    # 9. 导出结果
    print("步骤 9: 导出结果...")
    output_dir = "simulation_results"
    simulator.export_results(results, output_dir)
    print()
    
    # 10. 可视化
    print("步骤 10: 生成可视化...")
    visualize_results(results, output_dir)
    print()
    
    print("=" * 60)
    print("仿真完成！")
    print("=" * 60)
    
    return results


def visualize_results(results, output_dir):
    """可视化结果"""
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # 绘制水深分布
        if results.hydrodynamics is not None:
            hydro = results.hydrodynamics
            time_idx = len(hydro.times) // 2  # 中间时刻
            
            axes[0].plot(hydro.stations / 1000, hydro.depths[time_idx], 'b-', linewidth=2)
            axes[0].set_xlabel('桩号 (km)')
            axes[0].set_ylabel('水深 (m)')
            axes[0].set_title(f'水深分布 (t={hydro.times[time_idx]/3600:.1f}h)')
            axes[0].grid(True)
        
        # 绘制水质浓度
        if results.water_quality is not None:
            wq = results.water_quality
            time_idx = len(wq.times) // 2
            
            if 'DO' in wq.concentrations:
                axes[1].plot(
                    wq.stations / 1000,
                    wq.concentrations['DO'][time_idx],
                    'g-',
                    linewidth=2,
                    label='DO'
                )
            if 'NH3N' in wq.concentrations:
                axes[1].plot(
                    wq.stations / 1000,
                    wq.concentrations['NH3N'][time_idx],
                    'r-',
                    linewidth=2,
                    label='NH3-N'
                )
            
            axes[1].set_xlabel('桩号 (km)')
            axes[1].set_ylabel('浓度 (mg/L)')
            axes[1].set_title(f'水质分布 (t={wq.times[time_idx]/3600:.1f}h)')
            axes[1].legend()
            axes[1].grid(True)
        
        # 绘制稳定性系数
        if results.slope_stability is not None:
            stab = results.slope_stability
            time_idx = len(stab.times) // 2
            
            axes[2].plot(
                stab.stations / 1000,
                stab.comprehensive_factors[time_idx],
                'k-',
                linewidth=2,
                label='综合稳定系数'
            )
            axes[2].axhline(y=1.3, color='r', linestyle='--', label='安全阈值')
            axes[2].set_xlabel('桩号 (km)')
            axes[2].set_ylabel('稳定系数')
            axes[2].set_title(f'边坡稳定性分布 (t={stab.times[time_idx]/3600:.1f}h)')
            axes[2].legend()
            axes[2].grid(True)
        
        plt.tight_layout()
        fig.savefig(f"{output_dir}/visualization.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  可视化已保存: {output_dir}/visualization.png")
        
    except Exception as e:
        print(f"  可视化失败: {e}")


if __name__ == "__main__":
    results = run_example_simulation()
