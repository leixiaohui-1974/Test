"""
水动力学模块最小单元测试

测试目标：
1. 验证断面几何计算
2. 验证Preissmann求解器基本功能
3. 验证边界条件设置
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def test_section_geometry():
    """测试断面几何计算"""
    print("\n" + "="*60)
    print("测试1: 断面几何计算")
    print("="*60)
    
    from channel_stability.core.channel_system import ChannelSection, SoilProperties
    
    # 创建简单矩形断面
    section = ChannelSection(
        station=0.0,
        bed_elevation=100.0,
        bottom_width=5.0,
        side_slope=2.0,  # 1:2边坡
        max_depth=4.0,
        manning_n=0.025,
        slope_height=3.0,
        slope_angle=26.57,
        soil_properties=SoilProperties(
            cohesion=15.0,
            friction_angle=30.0,
            unit_weight=18.0,
            saturated_unit_weight=20.0,
            permeability=1e-5,
            porosity=0.35,
        ),
    )
    
    # 测试不同水深的几何参数
    depths = [0.5, 1.0, 2.0, 3.0]
    
    print("\n水深 (m) | 面积 (m²) | 湿周 (m) | 水力半径 (m) | 水面宽 (m)")
    print("-" * 70)
    
    for depth in depths:
        area = section.area(depth)
        wp = section.wetted_perimeter(depth)
        R = section.hydraulic_radius(depth)
        B = section.top_width(depth)
        
        print(f"{depth:8.2f} | {area:9.2f} | {wp:8.2f} | {R:12.3f} | {B:10.2f}")
        
        # 验证
        assert area > 0, f"面积应为正: {area}"
        assert wp > 0, f"湿周应为正: {wp}"
        assert R > 0, f"水力半径应为正: {R}"
        assert B >= section.bottom_width, f"水面宽应≥底宽: {B} < {section.bottom_width}"
    
    print("\n✓ 断面几何计算测试通过")
    return True


def test_preissmann_simple_channel():
    """测试Preissmann求解器 - 简单渠道"""
    print("\n" + "="*60)
    print("测试2: Preissmann求解器 - 简单恒定流")
    print("="*60)
    
    from channel_stability.core.channel_system import ChannelSystem, ChannelSection, SoilProperties
    from channel_stability.hydrodynamics.boundary_conditions import BoundaryConditions
    from channel_stability.hydrodynamics.preissmann_solver import PreissmannHydrodynamicSolver
    
    # 创建简单的5个断面渠道
    soil = SoilProperties(
        cohesion=15.0, friction_angle=30.0, unit_weight=18.0,
        saturated_unit_weight=20.0, permeability=1e-5, porosity=0.35
    )
    
    channel = ChannelSystem(
        name="简单测试渠道",
        total_length=1000.0,  # 1km
        sections=[],
    )
    
    # 创建5个断面
    for i in range(5):
        station = i * 250.0
        section = ChannelSection(
            station=station,
            bed_elevation=100.0 - 0.0001 * station,  # 万分之一坡度
            bottom_width=5.0,
            side_slope=2.0,
            max_depth=4.0,
            manning_n=0.025,
            slope_height=3.0,
            slope_angle=26.57,
            soil_properties=soil,
        )
        channel.sections.append(section)
    
    # 恒定边界条件
    bc = BoundaryConditions.create_constant_bc(
        upstream_q=10.0,  # 10 m³/s
        downstream_h=2.0,  # 2 m水深
    )
    
    # 求解器
    solver = PreissmannHydrodynamicSolver(
        channel=channel,
        boundary_conditions=bc,
        enable_adaptive_mesh=False,
    )
    
    # 运行短时间仿真（使用更小的时间步长以保证稳定性）
    print("\n运行仿真: 300秒 (5分钟), 时间步长=5秒")
    results = solver.solve(
        total_time=300.0,
        dt=5.0,
        initial_depth=2.0,
        initial_discharge=10.0,
        save_interval=10,
    )
    
    print(f"\n结果统计:")
    print(f"  时间步数: {len(results.times)}")
    print(f"  断面数: {len(results.stations)}")
    print(f"  最大水深: {np.max(results.depths):.3f} m")
    print(f"  最小水深: {np.min(results.depths):.3f} m")
    print(f"  最大流速: {np.max(results.velocities):.3f} m/s")
    print(f"  最大Froude数: {np.max(results.froude_numbers):.3f}")
    
    # 验证
    assert len(results.times) > 0, "应有时间步"
    assert len(results.stations) == 5, "应有5个断面"
    assert np.all(results.depths > 0), "所有水深应为正"
    assert np.all(np.isfinite(results.depths)), "水深应为有限值"
    assert np.all(results.froude_numbers >= 0), "Froude数应非负"
    
    # 检查恒定流是否收敛到稳态
    final_depths = results.depths[-1]
    print(f"\n最终水深分布: {final_depths}")
    depth_variation = np.std(final_depths)
    print(f"水深标准差: {depth_variation:.4f} m")
    
    # 检查数值合理性
    assert np.max(results.depths) < 10.0, f"水深过大: {np.max(results.depths)}"
    assert np.max(results.velocities) < 10.0, f"流速过大: {np.max(results.velocities)}"
    
    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 水深分布
    axes[0, 0].plot(results.stations, results.depths[0], 'b--', label='初始', alpha=0.5)
    axes[0, 0].plot(results.stations, results.depths[-1], 'r-', label='最终', linewidth=2)
    axes[0, 0].set_xlabel('桩号 (m)')
    axes[0, 0].set_ylabel('水深 (m)')
    axes[0, 0].set_title('水深分布')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 流速分布
    axes[0, 1].plot(results.stations, results.velocities[-1], 'g-', linewidth=2)
    axes[0, 1].set_xlabel('桩号 (m)')
    axes[0, 1].set_ylabel('流速 (m/s)')
    axes[0, 1].set_title('流速分布')
    axes[0, 1].grid(True)
    
    # 时间历程（中间断面）
    mid_idx = len(results.stations) // 2
    axes[1, 0].plot(results.times / 60, results.depths[:, mid_idx], 'b-', linewidth=2)
    axes[1, 0].set_xlabel('时间 (分钟)')
    axes[1, 0].set_ylabel('水深 (m)')
    axes[1, 0].set_title(f'中间断面水深时间历程 (x={results.stations[mid_idx]:.0f}m)')
    axes[1, 0].grid(True)
    
    # Froude数
    axes[1, 1].plot(results.stations, results.froude_numbers[-1], 'orange', linewidth=2)
    axes[1, 1].axhline(y=1.0, color='r', linestyle='--', label='临界流')
    axes[1, 1].set_xlabel('桩号 (m)')
    axes[1, 1].set_ylabel('Froude数')
    axes[1, 1].set_title('Froude数分布')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('test_hydrodynamics_results.png', dpi=150, bbox_inches='tight')
    print("\n✓ 结果已保存: test_hydrodynamics_results.png")
    plt.close()
    
    print("\n✓ Preissmann求解器测试通过")
    return results


def test_boundary_conditions():
    """测试边界条件"""
    print("\n" + "="*60)
    print("测试3: 边界条件设置")
    print("="*60)
    
    from channel_stability.hydrodynamics.boundary_conditions import BoundaryConditions
    
    # 测试恒定边界条件
    bc1 = BoundaryConditions.create_constant_bc(
        upstream_q=10.0,
        downstream_h=2.0,
    )
    
    q = bc1.get_upstream_discharge(100.0)
    h = bc1.get_downstream_stage(100.0, 10.0)
    
    print(f"\n恒定边界条件:")
    print(f"  上游流量: {q:.2f} m³/s")
    print(f"  下游水位: {h:.2f} m")
    
    assert abs(q - 10.0) < 1e-6, "上游流量应为10.0"
    assert abs(h - 2.0) < 1e-6, "下游水位应为2.0"
    
    # 测试流量过程
    time_series = [(0, 5.0), (300, 15.0), (600, 5.0)]
    bc2 = BoundaryConditions.create_hydrograph_bc(
        time_series=time_series,
        downstream_h=2.0,
    )
    
    print(f"\n流量过程边界条件:")
    for t in [0, 150, 300, 450, 600]:
        q = bc2.get_upstream_discharge(t)
        print(f"  t={t:3d}s: Q={q:.2f} m³/s")
    
    assert abs(bc2.get_upstream_discharge(0) - 5.0) < 1e-6, "t=0时流量应为5.0"
    assert abs(bc2.get_upstream_discharge(300) - 15.0) < 1e-6, "t=300时流量应为15.0"
    
    print("\n✓ 边界条件测试通过")
    return True


def run_all_hydrodynamics_tests():
    """运行所有水动力学测试"""
    print("\n" + "#"*60)
    print("# 水动力学模块单元测试")
    print("#"*60)
    
    try:
        # 测试1: 断面几何
        test_section_geometry()
        
        # 测试2: Preissmann求解器
        results = test_preissmann_simple_channel()
        
        # 测试3: 边界条件
        test_boundary_conditions()
        
        print("\n" + "#"*60)
        print("# 水动力学模块测试 - 全部通过 ✓")
        print("#"*60)
        return True, results
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


if __name__ == "__main__":
    success, results = run_all_hydrodynamics_tests()
    exit(0 if success else 1)
