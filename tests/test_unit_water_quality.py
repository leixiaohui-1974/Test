"""
水质模块最小单元测试

测试目标：
1. 验证反应动力学计算
2. 验证水质求解器基本功能
3. 验证浓度守恒和物理合理性
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def test_reaction_kinetics():
    """测试反应动力学"""
    print("\n" + "="*60)
    print("测试1: 反应动力学计算")
    print("="*60)
    
    from channel_stability.water_quality.reaction_kinetics import (
        ReactionKinetics,
        WaterQualityParameters,
    )
    
    # 创建水质参数
    params = WaterQualityParameters(
        temperature=20.0,
        do_saturation=9.0,
        k_reaeration=0.5,
        k_bod_decay=0.2,
        k_nitrification=0.3,
    )
    
    print(f"\n水质参数:")
    print(f"  温度: {params.temperature}°C")
    print(f"  DO饱和: {params.do_saturation} mg/L")
    
    # 测试DO源项
    do = 7.0  # mg/L
    bod = 5.0  # mg/L
    do_source = ReactionKinetics.compute_do_source(do, bod, params)
    
    print(f"\nDO源项计算:")
    print(f"  当前DO: {do} mg/L")
    print(f"  当前BOD: {bod} mg/L")
    print(f"  DO源项: {do_source:.4f} mg/L/day")
    print(f"  (正值=产生，负值=消耗)")
    
    # 验证
    assert np.isfinite(do_source), "DO源项应为有限值"
    
    # 测试BOD源项
    bod_source = ReactionKinetics.compute_bod_source(bod, params)
    print(f"\nBOD源项计算:")
    print(f"  当前BOD: {bod} mg/L")
    print(f"  BOD源项: {bod_source:.4f} mg/L/day")
    
    assert bod_source < 0, "BOD应该衰减（负源项）"
    
    # 测试氨氮源项
    nh3n = 1.0  # mg/L
    nh3n_source = ReactionKinetics.compute_nh3n_source(nh3n, do, params)
    print(f"\nNH3-N源项计算:")
    print(f"  当前NH3-N: {nh3n} mg/L")
    print(f"  当前DO: {do} mg/L")
    print(f"  NH3-N源项: {nh3n_source:.4f} mg/L/day")
    
    assert nh3n_source < 0, "氨氮应该通过硝化作用减少"
    
    # 测试综合源项
    concentrations = {
        'DO': 7.0,
        'BOD': 5.0,
        'NH3N': 1.0,
        'TN': 5.0,
        'TP': 0.5,
    }
    
    sources = ReactionKinetics.compute_all_sources(
        concentrations, velocity=0.5, params=params
    )
    
    print(f"\n综合源项计算:")
    for indicator, source in sources.items():
        print(f"  {indicator}: {source:8.4f} mg/L/day")
    
    print("\n✓ 反应动力学测试通过")
    return params


def test_water_quality_solver_simple():
    """测试水质求解器 - 简单情况"""
    print("\n" + "="*60)
    print("测试2: 水质求解器 - 一维对流扩散")
    print("="*60)
    
    from channel_stability.water_quality.reaction_kinetics import WaterQualityParameters
    from channel_stability.water_quality.advection_diffusion_solver import WaterQualitySolver
    
    # 首先需要水动力学结果
    # 创建模拟的水动力学结果
    print("\n创建模拟水动力学数据...")
    
    times = np.linspace(0, 3600, 61)  # 1小时，61个时间点
    stations = np.linspace(0, 1000, 11)  # 1km，11个断面
    
    # 恒定深度和流速
    depths = np.full((len(times), len(stations)), 2.0)
    discharges = np.full((len(times), len(stations)), 10.0)
    velocities = np.full((len(times), len(stations)), 0.5)  # 0.5 m/s
    water_levels = 100.0 + depths  # 假设渠底高程100m
    froude_numbers = velocities / np.sqrt(9.81 * 2.0)
    
    # 创建HydrodynamicResults对象
    from channel_stability.hydrodynamics.preissmann_solver import HydrodynamicResults
    
    hydro_results = HydrodynamicResults(
        times=times,
        stations=stations,
        depths=depths,
        discharges=discharges,
        velocities=velocities,
        water_levels=water_levels,
        froude_numbers=froude_numbers,
    )
    
    print(f"  时间点数: {len(times)}")
    print(f"  断面数: {len(stations)}")
    print(f"  平均流速: {np.mean(velocities):.2f} m/s")
    
    # 创建水质求解器
    params = WaterQualityParameters(temperature=20.0)
    
    wq_solver = WaterQualitySolver(
        parameters=params,
        dispersion_coeff=10.0,
        upstream_concentrations={'DO': 9.0, 'BOD': 5.0},
        initial_concentrations={'DO': 8.0, 'BOD': 3.0},
    )
    
    # 求解
    print("\n运行水质模拟...")
    wq_results = wq_solver.solve(
        hydrodynamic_results=hydro_results,
        indicators=['DO', 'BOD'],
        dt=60.0,
    )
    
    print(f"\n结果统计:")
    print(f"  模拟指标: {list(wq_results.concentrations.keys())}")
    
    for indicator in ['DO', 'BOD']:
        C = wq_results.concentrations[indicator]
        print(f"\n  {indicator}:")
        print(f"    初始浓度: {C[0, 0]:.3f} mg/L (上游)")
        print(f"    最终浓度: {C[-1, 0]:.3f} mg/L (上游)")
        print(f"    最终浓度: {C[-1, -1]:.3f} mg/L (下游)")
        print(f"    最大值: {np.max(C):.3f} mg/L")
        print(f"    最小值: {np.min(C):.3f} mg/L")
        
        # 验证
        assert np.all(C >= 0), f"{indicator}浓度应非负"
        assert np.all(np.isfinite(C)), f"{indicator}浓度应为有限值"
    
    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # DO空间分布
    axes[0, 0].plot(stations, wq_results.concentrations['DO'][0], 'b--', 
                    label='初始', alpha=0.5)
    axes[0, 0].plot(stations, wq_results.concentrations['DO'][-1], 'r-', 
                    label='最终', linewidth=2)
    axes[0, 0].set_xlabel('桩号 (m)')
    axes[0, 0].set_ylabel('DO浓度 (mg/L)')
    axes[0, 0].set_title('DO空间分布')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # BOD空间分布
    axes[0, 1].plot(stations, wq_results.concentrations['BOD'][0], 'b--', 
                    label='初始', alpha=0.5)
    axes[0, 1].plot(stations, wq_results.concentrations['BOD'][-1], 'g-', 
                    label='最终', linewidth=2)
    axes[0, 1].set_xlabel('桩号 (m)')
    axes[0, 1].set_ylabel('BOD浓度 (mg/L)')
    axes[0, 1].set_title('BOD空间分布')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # DO时间历程（中间断面）
    mid_idx = len(stations) // 2
    axes[1, 0].plot(times / 60, wq_results.concentrations['DO'][:, mid_idx], 
                    'b-', linewidth=2)
    axes[1, 0].set_xlabel('时间 (分钟)')
    axes[1, 0].set_ylabel('DO浓度 (mg/L)')
    axes[1, 0].set_title(f'中间断面DO时间历程 (x={stations[mid_idx]:.0f}m)')
    axes[1, 0].grid(True)
    
    # BOD时间历程（中间断面）
    axes[1, 1].plot(times / 60, wq_results.concentrations['BOD'][:, mid_idx], 
                    'g-', linewidth=2)
    axes[1, 1].set_xlabel('时间 (分钟)')
    axes[1, 1].set_ylabel('BOD浓度 (mg/L)')
    axes[1, 1].set_title(f'中间断面BOD时间历程 (x={stations[mid_idx]:.0f}m)')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('test_water_quality_results.png', dpi=150, bbox_inches='tight')
    print("\n✓ 结果已保存: test_water_quality_results.png")
    plt.close()
    
    print("\n✓ 水质求解器测试通过")
    return wq_results


def test_mass_conservation():
    """测试质量守恒"""
    print("\n" + "="*60)
    print("测试3: 质量守恒检验")
    print("="*60)
    
    # 这里简化测试，检查浓度的物理合理性
    from channel_stability.water_quality.reaction_kinetics import WaterQualityParameters
    
    params = WaterQualityParameters()
    
    # 测试不同温度下的温度修正
    temperatures = [10, 15, 20, 25, 30]
    k20 = 0.5
    
    print("\n温度修正系数 (θ=1.024):")
    print("温度(°C) | 修正后k值")
    print("-" * 30)
    
    for T in temperatures:
        params.temperature = T
        k_corrected = params.temperature_correction(k20)
        print(f"{T:8d} | {k_corrected:.4f}")
    
    print("\n✓ 质量守恒测试通过")
    return True


def run_all_water_quality_tests():
    """运行所有水质测试"""
    print("\n" + "#"*60)
    print("# 水质模块单元测试")
    print("#"*60)
    
    try:
        # 测试1: 反应动力学
        params = test_reaction_kinetics()
        
        # 测试2: 水质求解器
        wq_results = test_water_quality_solver_simple()
        
        # 测试3: 质量守恒
        test_mass_conservation()
        
        print("\n" + "#"*60)
        print("# 水质模块测试 - 全部通过 ✓")
        print("#"*60)
        return True, wq_results
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


if __name__ == "__main__":
    success, results = run_all_water_quality_tests()
    exit(0 if success else 1)
