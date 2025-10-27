"""
运行所有单元测试（简化版，无需额外依赖）

直接导入和测试核心功能
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_hydrodynamics_module():
    """测试水动力学模块"""
    print("\n" + "="*70)
    print("测试模块 1: 水动力学")
    print("="*70)
    
    try:
        # 测试导入
        from channel_stability.core.channel_system import (
            ChannelSystem, ChannelSection, SoilProperties, LiningProperties
        )
        from channel_stability.hydrodynamics.boundary_conditions import BoundaryConditions
        from channel_stability.hydrodynamics.preissmann_solver import PreissmannHydrodynamicSolver
        print("✓ 模块导入成功")
        
        # 测试断面几何
        soil = SoilProperties(
            cohesion=15.0, friction_angle=30.0, unit_weight=18.0,
            saturated_unit_weight=20.0, permeability=1e-5, porosity=0.35
        )
        
        section = ChannelSection(
            station=0.0,
            bed_elevation=100.0,
            bottom_width=5.0,
            side_slope=2.0,
            max_depth=4.0,
            manning_n=0.025,
            slope_height=3.0,
            slope_angle=26.57,
            soil_properties=soil,
        )
        
        # 测试几何计算
        depth = 2.0
        area = section.area(depth)
        wp = section.wetted_perimeter(depth)
        R = section.hydraulic_radius(depth)
        B = section.top_width(depth)
        
        print(f"✓ 断面几何计算: 水深={depth}m, 面积={area:.2f}m², 水力半径={R:.3f}m")
        
        assert area > 0, "面积应为正"
        assert R > 0, "水力半径应为正"
        assert B >= section.bottom_width, "水面宽应≥底宽"
        
        # 测试渠道系统
        channel = ChannelSystem(
            name="测试渠道",
            total_length=1000.0,
            sections=[],
        )
        
        for i in range(5):
            s = ChannelSection(
                station=i * 250.0,
                bed_elevation=100.0 - 0.0001 * i * 250.0,
                bottom_width=5.0,
                side_slope=2.0,
                max_depth=4.0,
                manning_n=0.025,
                slope_height=3.0,
                slope_angle=26.57,
                soil_properties=soil,
            )
            channel.sections.append(s)
        
        print(f"✓ 渠道系统创建: {channel.num_sections}个断面, 长度{channel.total_length}m")
        
        # 测试边界条件
        bc = BoundaryConditions.create_constant_bc(
            upstream_q=10.0,
            downstream_h=2.0,
        )
        
        q_test = bc.get_upstream_discharge(100.0)
        h_test = bc.get_downstream_stage(100.0, 10.0)
        
        print(f"✓ 边界条件设置: 上游流量={q_test}m³/s, 下游水位={h_test}m")
        
        assert abs(q_test - 10.0) < 1e-6, "上游流量应为10.0"
        assert abs(h_test - 2.0) < 1e-6, "下游水位应为2.0"
        
        # 测试求解器（快速测试）
        solver = PreissmannHydrodynamicSolver(
            channel=channel,
            boundary_conditions=bc,
            enable_adaptive_mesh=False,
        )
        
        print("✓ Preissmann求解器初始化成功")
        
        # 运行很短的仿真
        print("  运行快速仿真测试 (60秒)...")
        results = solver.solve(
            total_time=60.0,
            dt=10.0,
            initial_depth=2.0,
            initial_discharge=10.0,
            save_interval=10,
        )
        
        print(f"✓ 仿真完成: {len(results.times)}个时间步, {len(results.stations)}个断面")
        print(f"  最大水深: {results.depths.max():.3f}m")
        print(f"  最大流速: {results.velocities.max():.3f}m/s")
        
        return True
        
    except Exception as e:
        print(f"✗ 水动力学测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_water_quality_module():
    """测试水质模块"""
    print("\n" + "="*70)
    print("测试模块 2: 水质")
    print("="*70)
    
    try:
        # 测试导入
        from channel_stability.water_quality.reaction_kinetics import (
            ReactionKinetics, WaterQualityParameters
        )
        from channel_stability.water_quality.advection_diffusion_solver import WaterQualitySolver
        print("✓ 模块导入成功")
        
        # 测试反应动力学
        params = WaterQualityParameters(
            temperature=20.0,
            do_saturation=9.0,
            k_reaeration=0.5,
            k_bod_decay=0.2,
        )
        
        print(f"✓ 水质参数设置: 温度={params.temperature}°C, DO饱和={params.do_saturation}mg/L")
        
        # 测试DO源项
        do = 7.0
        bod = 5.0
        do_source = ReactionKinetics.compute_do_source(do, bod, params)
        
        print(f"✓ DO源项计算: 当前DO={do}mg/L, BOD={bod}mg/L, 源项={do_source:.4f}mg/L/day")
        
        assert do_source is not None, "DO源项计算失败"
        
        # 测试BOD源项
        bod_source = ReactionKinetics.compute_bod_source(bod, params)
        print(f"✓ BOD源项计算: 源项={bod_source:.4f}mg/L/day (应为负)")
        
        assert bod_source < 0, "BOD应衰减"
        
        # 测试综合源项
        concentrations = {'DO': 7.0, 'BOD': 5.0, 'NH3N': 1.0, 'TN': 5.0, 'TP': 0.5}
        sources = ReactionKinetics.compute_all_sources(concentrations, 0.5, params)
        
        print(f"✓ 综合源项计算: {len(sources)}个指标")
        for ind, src in sources.items():
            print(f"  {ind}: {src:8.4f} mg/L/day")
        
        # 测试水质求解器（不实际运行，只测试初始化）
        wq_solver = WaterQualitySolver(parameters=params)
        print("✓ 水质求解器初始化成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 水质测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_slope_stability_module():
    """测试边坡稳定性模块"""
    print("\n" + "="*70)
    print("测试模块 3: 边坡稳定性")
    print("="*70)
    
    try:
        # 测试导入
        from channel_stability.core.channel_system import SoilProperties, LiningProperties
        from channel_stability.slope_stability.failure_mechanisms import (
            LiningStabilityAnalysis, StabilityFactors
        )
        from channel_stability.slope_stability.stability_calculator import SlopeStabilityCalculator
        print("✓ 模块导入成功")
        
        # 创建土壤和衬砌特性
        soil = SoilProperties(
            cohesion=15.0,
            friction_angle=30.0,
            unit_weight=18.0,
            saturated_unit_weight=20.0,
            permeability=1e-5,
            porosity=0.35,
        )
        
        lining = LiningProperties(
            thickness=0.1,
            density=2400.0,
            elastic_modulus=30.0,
            friction_coeff=0.4,
        )
        
        print(f"✓ 土壤参数: 粘聚力={soil.cohesion}kPa, 摩擦角={soil.friction_angle}°")
        print(f"✓ 衬砌参数: 厚度={lining.thickness}m, 密度={lining.density}kg/m³")
        
        # 测试稳定性计算
        slope_angle = 26.57
        slope_height = 3.0
        groundwater_depth = 1.0
        channel_water_level = 2.0
        rainfall = 0.0
        
        factors = LiningStabilityAnalysis.compute_comprehensive_stability(
            slope_angle=slope_angle,
            slope_height=slope_height,
            soil_props=soil,
            lining_props=lining,
            groundwater_depth=groundwater_depth,
            channel_water_level=channel_water_level,
            rainfall_intensity=rainfall,
        )
        
        print(f"✓ 稳定性计算完成:")
        print(f"  滑动系数: {factors.sliding_factor:.2f}")
        print(f"  倾覆系数: {factors.overturning_factor:.2f}")
        print(f"  浮托系数: {factors.uplift_factor:.2f}")
        print(f"  渗透系数: {factors.seepage_factor:.2f}")
        print(f"  综合系数: {factors.comprehensive_factor:.2f}")
        print(f"  是否稳定: {'是' if factors.is_stable else '否'}")
        
        assert factors.sliding_factor > 0, "滑动系数应为正"
        assert factors.comprehensive_factor > 0, "综合系数应为正"
        
        # 测试不同地下水位
        print("\n✓ 测试不同地下水位:")
        for gw_d in [0.0, 1.0, 2.0, 3.0]:
            f = LiningStabilityAnalysis.compute_comprehensive_stability(
                slope_angle, slope_height, soil, lining,
                gw_d, channel_water_level, rainfall
            )
            print(f"  地下水埋深={gw_d:.1f}m: 综合系数={f.comprehensive_factor:.2f}, "
                  f"{'稳定' if f.is_stable else '不稳定'}")
        
        return True
        
    except Exception as e:
        print(f"✗ 边坡稳定性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "#"*70)
    print("# 明渠边坡稳定性系统 - 单元测试套件")
    print("#"*70)
    
    results = {}
    
    # 测试水动力学
    results['hydrodynamics'] = test_hydrodynamics_module()
    
    # 测试水质
    results['water_quality'] = test_water_quality_module()
    
    # 测试边坡稳定性
    results['slope_stability'] = test_slope_stability_module()
    
    # 汇总结果
    print("\n" + "#"*70)
    print("# 测试结果汇总")
    print("#"*70)
    
    for module, success in results.items():
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{module:20s}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "="*70)
        print("所有单元测试通过！ ✓✓✓")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("部分测试失败！")
        print("="*70)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
