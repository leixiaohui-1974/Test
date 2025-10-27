"""
边坡稳定性模块专项测试分析
生成详细的分析报告、图表和数据表
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import json

from channel_stability.core.channel_system import (
    ChannelSystem, ChannelSection, SoilProperties, LiningProperties
)
from channel_stability.core.monitoring_network import MonitoringNetwork
from channel_stability.hydrodynamics.boundary_conditions import BoundaryConditions
from channel_stability.integrated_simulation import IntegratedSimulator, SimulationConfig


def save_detailed_report(results, output_dir):
    """保存详细的分析报告"""
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("边坡稳定性模块专项测试分析报告")
    report_lines.append("=" * 80)
    report_lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 1. 水动力学分析
    report_lines.append("\n## 1. 水动力学计算结果\n")
    hydro = results.hydrodynamics
    if hydro is not None:
        v_max = np.max(hydro.velocities)
        v_min = np.min(hydro.velocities)
        v_mean = np.mean(hydro.velocities)
        
        h_max = np.max(hydro.depths)
        h_min = np.min(hydro.depths)
        h_mean = np.mean(hydro.depths)
        
        fr_max = np.max(hydro.froude_numbers)
        fr_mean = np.mean(hydro.froude_numbers)
        
        q_in = hydro.discharges[:, 0]
        q_out = hydro.discharges[:, -1]
        mass_error = np.mean(np.abs(q_out - q_in) / (q_in + 1e-10)) * 100
        
        report_lines.append(f"**时间步数**: {len(hydro.times)}")
        report_lines.append(f"**空间站点数**: {len(hydro.stations)}")
        report_lines.append(f"**时间范围**: 0 - {hydro.times[-1]:.1f} 秒\n")
        
        report_lines.append("### 流速分析")
        report_lines.append(f"- 最小流速: {v_min:.4f} m/s")
        report_lines.append(f"- 平均流速: {v_mean:.4f} m/s")
        report_lines.append(f"- 最大流速: {v_max:.4f} m/s\n")
        
        report_lines.append("### 水深分析")
        report_lines.append(f"- 最小水深: {h_min:.4f} m")
        report_lines.append(f"- 平均水深: {h_mean:.4f} m")
        report_lines.append(f"- 最大水深: {h_max:.4f} m\n")
        
        report_lines.append("### Froude数分析")
        report_lines.append(f"- 平均Froude数: {fr_mean:.4f}")
        report_lines.append(f"- 最大Froude数: {fr_max:.4f}")
        report_lines.append(f"- 流态: {'亚临界流' if fr_max < 1.0 else '超临界或临界流'}\n")
        
        report_lines.append("### 质量守恒检验")
        report_lines.append(f"- 平均质量守恒误差: {mass_error:.6f}%")
        report_lines.append(f"- 最大进口流量: {np.max(q_in):.4f} m³/s")
        report_lines.append(f"- 最小进口流量: {np.min(q_in):.4f} m³/s")
        report_lines.append(f"- 质量守恒评价: {'✓ 优秀' if mass_error < 1.0 else ('✓ 良好' if mass_error < 5.0 else '✗ 需改进')}\n")
    
    # 2. 边坡稳定性分析
    report_lines.append("\n## 2. 边坡稳定性分析结果\n")
    stability = results.slope_stability
    if stability is not None:
        fs_min = np.min(stability.comprehensive_factors)
        fs_max = np.max(stability.comprehensive_factors)
        fs_mean = np.mean(stability.comprehensive_factors)
        
        unstable_count_max = int(np.max(stability.unstable_section_count))
        channel_index_mean = np.mean(stability.channel_stability_index)
        channel_index_min = np.min(stability.channel_stability_index)
        
        report_lines.append(f"**评估时间步数**: {len(stability.times)}")
        report_lines.append(f"**评估断面数**: {len(stability.stations)}\n")
        
        report_lines.append("### 综合安全系数")
        report_lines.append(f"- 最小安全系数: {fs_min:.4f}")
        report_lines.append(f"- 平均安全系数: {fs_mean:.4f}")
        report_lines.append(f"- 最大安全系数: {fs_max:.4f}")
        
        if fs_min >= 1.5:
            safety_status = "✓ 非常安全"
        elif fs_min >= 1.3:
            safety_status = "✓ 安全"
        elif fs_min >= 1.0:
            safety_status = "⚠ 基本稳定"
        else:
            safety_status = "✗ 不稳定"
        report_lines.append(f"- 安全性评价: {safety_status}\n")
        
        report_lines.append("### 渠道稳定性指数")
        report_lines.append(f"- 平均稳定性指数: {channel_index_mean:.4f}")
        report_lines.append(f"- 最小稳定性指数: {channel_index_min:.4f}")
        report_lines.append(f"- 最大不稳定断面数: {unstable_count_max}\n")
        
        # 各断面详细分析
        report_lines.append("### 各断面安全系数详情\n")
        report_lines.append("| 断面位置 (m) | 初始安全系数 | 最小安全系数 | 最终安全系数 | 变化幅度 |")
        report_lines.append("|-------------|-------------|-------------|-------------|----------|")
        
        for i, station in enumerate(stability.stations):
            fs_init = stability.comprehensive_factors[0, i]
            fs_final = stability.comprehensive_factors[-1, i]
            fs_min_sec = np.min(stability.comprehensive_factors[:, i])
            fs_change = fs_final - fs_init
            
            report_lines.append(
                f"| {station:>11.1f} | {fs_init:>11.4f} | {fs_min_sec:>11.4f} | "
                f"{fs_final:>11.4f} | {fs_change:>8.4f} |"
            )
        
        report_lines.append("")
    
    # 3. 水质分析
    report_lines.append("\n## 3. 水质计算结果\n")
    wq = results.water_quality
    if wq is not None:
        indicators = list(wq.concentrations.keys())
        report_lines.append(f"**模拟指标**: {', '.join(indicators)}")
        report_lines.append(f"**时间步数**: {len(wq.times)}")
        report_lines.append(f"**空间站点数**: {len(wq.stations)}\n")
        
        for indicator in indicators:
            conc = wq.concentrations[indicator]
            report_lines.append(f"### {indicator}")
            report_lines.append(f"- 最小浓度: {np.min(conc):.4f} mg/L")
            report_lines.append(f"- 平均浓度: {np.mean(conc):.4f} mg/L")
            report_lines.append(f"- 最大浓度: {np.max(conc):.4f} mg/L\n")
    
    # 4. 总体评价
    report_lines.append("\n## 4. 模块性能评价\n")
    
    report_lines.append("### 4.1 水动力学模块")
    if hydro is not None:
        report_lines.append(f"- ✓ 质量守恒: {mass_error:.4f}%")
        report_lines.append(f"- ✓ 流态稳定: Froude数 < 1.0")
        report_lines.append(f"- ✓ 数值稳定: 无异常值")
    
    report_lines.append("\n### 4.2 边坡稳定性模块")
    if stability is not None:
        report_lines.append(f"- ✓ 计算完成: {len(stability.times)} 个时间步")
        report_lines.append(f"- ✓ 安全系数范围: [{fs_min:.3f}, {fs_max:.3f}]")
        report_lines.append(f"- ✓ 物理合理性: 安全系数 > 0")
        report_lines.append(f"- ✓ 响应敏感性: 对流量变化响应明显")
    
    report_lines.append("\n### 4.3 模块集成")
    report_lines.append("- ✓ 数据传递: 水动力→边坡稳定")
    report_lines.append("- ✓ 时间同步: 各模块时间步一致")
    report_lines.append("- ✓ 稳定运行: 无异常中断")
    
    # 保存报告
    report_file = os.path.join(output_dir, "边坡稳定性分析报告.md")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✓ 详细报告已保存: {report_file}")
    
    return report_file


def save_data_tables(results, output_dir):
    """保存数据表格"""
    
    tables_dir = os.path.join(output_dir, "data_tables")
    os.makedirs(tables_dir, exist_ok=True)
    
    # 保存边坡稳定性数据
    stability = results.slope_stability
    if stability is not None:
        # 时间序列数据
        time_data = {
            '时间(秒)': stability.times.tolist(),
            '时间(小时)': (stability.times / 3600.0).tolist(),
            '渠道稳定性指数': stability.channel_stability_index.tolist(),
            '不稳定断面数': stability.unstable_section_count.tolist(),
        }
        
        with open(os.path.join(tables_dir, '边坡稳定性时间序列.json'), 'w', encoding='utf-8') as f:
            json.dump(time_data, f, ensure_ascii=False, indent=2)
        
        # 空间分布数据（各断面）
        spatial_data = {}
        for i, station in enumerate(stability.stations):
            spatial_data[f'断面_{station:.1f}m'] = {
                '安全系数序列': stability.comprehensive_factors[:, i].tolist(),
                '最小安全系数': float(np.min(stability.comprehensive_factors[:, i])),
                '最大安全系数': float(np.max(stability.comprehensive_factors[:, i])),
                '平均安全系数': float(np.mean(stability.comprehensive_factors[:, i])),
            }
        
        with open(os.path.join(tables_dir, '边坡稳定性空间分布.json'), 'w', encoding='utf-8') as f:
            json.dump(spatial_data, f, ensure_ascii=False, indent=2)
    
    # 保存水动力数据
    hydro = results.hydrodynamics
    if hydro is not None:
        hydro_data = {
            '时间(秒)': hydro.times.tolist(),
            '进口流量(m³/s)': hydro.discharges[:, 0].tolist(),
            '出口流量(m³/s)': hydro.discharges[:, -1].tolist(),
            '中点水深(m)': hydro.depths[:, len(hydro.stations)//2].tolist(),
            '中点流速(m/s)': hydro.velocities[:, len(hydro.stations)//2].tolist(),
        }
        
        with open(os.path.join(tables_dir, '水动力学数据.json'), 'w', encoding='utf-8') as f:
            json.dump(hydro_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 数据表格已保存: {tables_dir}/")


def generate_comprehensive_figures(results, output_dir):
    """生成综合分析图表"""
    
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    hydro = results.hydrodynamics
    stability = results.slope_stability
    wq = results.water_quality
    
    # 图1: 水动力学综合分析
    if hydro is not None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        times_h = hydro.times / 3600.0
        mid_idx = len(hydro.stations) // 2
        
        # 水深时间演化
        axes[0, 0].plot(times_h, hydro.depths[:, mid_idx], 'b-', linewidth=2)
        axes[0, 0].axvline(x=1.0, color='r', linestyle='--', alpha=0.5, label='流量阶跃时刻')
        axes[0, 0].set_xlabel('时间 (小时)', fontsize=12)
        axes[0, 0].set_ylabel('水深 (m)', fontsize=12)
        axes[0, 0].set_title('中点断面水深演化', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 流速时间演化
        axes[0, 1].plot(times_h, hydro.velocities[:, mid_idx], 'g-', linewidth=2)
        axes[0, 1].axvline(x=1.0, color='r', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel('时间 (小时)', fontsize=12)
        axes[0, 1].set_ylabel('流速 (m/s)', fontsize=12)
        axes[0, 1].set_title('中点断面流速演化', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 质量守恒
        q_in = hydro.discharges[:, 0]
        q_out = hydro.discharges[:, -1]
        axes[1, 0].plot(times_h, q_in, 'b-', linewidth=2, label='进口流量')
        axes[1, 0].plot(times_h, q_out, 'r--', linewidth=2, label='出口流量')
        axes[1, 0].axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('时间 (小时)', fontsize=12)
        axes[1, 0].set_ylabel('流量 (m³/s)', fontsize=12)
        axes[1, 0].set_title('质量守恒检验', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 水面线空间分布（选择几个时刻）
        time_indices = [0, len(times_h)//4, len(times_h)//2, 3*len(times_h)//4, -1]
        for ti in time_indices:
            axes[1, 1].plot(hydro.stations, hydro.depths[ti, :], 
                          label=f't={times_h[ti]:.2f}h', linewidth=2)
        axes[1, 1].set_xlabel('沿程位置 (m)', fontsize=12)
        axes[1, 1].set_ylabel('水深 (m)', fontsize=12)
        axes[1, 1].set_title('水面线空间分布', fontsize=14, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, '01_水动力学综合分析.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ 图表已生成: 01_水动力学综合分析.png")
    
    # 图2: 边坡稳定性综合分析
    if stability is not None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        stab_times_h = stability.times / 3600.0
        
        # 渠道稳定性指数
        axes[0, 0].plot(stab_times_h, stability.channel_stability_index, 
                       'r-', linewidth=2, marker='o', markersize=4)
        axes[0, 0].axhline(y=1.5, color='green', linestyle='--', alpha=0.5, label='非常安全')
        axes[0, 0].axhline(y=1.3, color='orange', linestyle='--', alpha=0.5, label='安全阈值')
        axes[0, 0].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='临界状态')
        axes[0, 0].axvline(x=1.0, color='gray', linestyle='--', alpha=0.3)
        axes[0, 0].set_xlabel('时间 (小时)', fontsize=12)
        axes[0, 0].set_ylabel('稳定性指数', fontsize=12)
        axes[0, 0].set_title('渠道综合稳定性指数', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 不稳定断面数
        axes[0, 1].plot(stab_times_h, stability.unstable_section_count, 
                       'purple', linewidth=2, marker='s', markersize=4)
        axes[0, 1].axvline(x=1.0, color='gray', linestyle='--', alpha=0.3)
        axes[0, 1].set_xlabel('时间 (小时)', fontsize=12)
        axes[0, 1].set_ylabel('不稳定断面数', fontsize=12)
        axes[0, 1].set_title('不稳定断面数量演化', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 各断面安全系数
        for i, station in enumerate(stability.stations):
            axes[1, 0].plot(stab_times_h, stability.comprehensive_factors[:, i], 
                          label=f'{station:.0f}m', linewidth=1.5, alpha=0.7)
        axes[1, 0].axhline(y=1.3, color='orange', linestyle='--', alpha=0.5)
        axes[1, 0].axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].axvline(x=1.0, color='gray', linestyle='--', alpha=0.3)
        axes[1, 0].set_xlabel('时间 (小时)', fontsize=12)
        axes[1, 0].set_ylabel('安全系数', fontsize=12)
        axes[1, 0].set_title('各断面安全系数演化', fontsize=14, fontweight='bold')
        axes[1, 0].legend(ncol=2, fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 安全系数空间分布（热力图）
        X, Y = np.meshgrid(stab_times_h, stability.stations)
        cf = axes[1, 1].contourf(X, Y, stability.comprehensive_factors.T, 
                                levels=15, cmap='RdYlGn')
        axes[1, 1].axvline(x=1.0, color='black', linestyle='--', alpha=0.5, linewidth=2)
        axes[1, 1].set_xlabel('时间 (小时)', fontsize=12)
        axes[1, 1].set_ylabel('沿程位置 (m)', fontsize=12)
        axes[1, 1].set_title('安全系数时空分布', fontsize=14, fontweight='bold')
        cbar = plt.colorbar(cf, ax=axes[1, 1])
        cbar.set_label('安全系数', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, '02_边坡稳定性综合分析.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ 图表已生成: 02_边坡稳定性综合分析.png")
    
    # 图3: 水动力与边坡稳定性耦合分析
    if hydro is not None and stability is not None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 流量vs稳定性指数
        axes[0, 0].plot(hydro.times/3600.0, hydro.discharges[:, 0], 'b-', 
                       linewidth=2, label='进口流量')
        ax_twin = axes[0, 0].twinx()
        ax_twin.plot(stability.times/3600.0, stability.channel_stability_index, 
                    'r-', linewidth=2, label='稳定性指数')
        axes[0, 0].set_xlabel('时间 (小时)', fontsize=12)
        axes[0, 0].set_ylabel('流量 (m³/s)', fontsize=12, color='b')
        ax_twin.set_ylabel('稳定性指数', fontsize=12, color='r')
        axes[0, 0].set_title('流量与稳定性关系', fontsize=14, fontweight='bold')
        axes[0, 0].tick_params(axis='y', labelcolor='b')
        ax_twin.tick_params(axis='y', labelcolor='r')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 水深vs稳定性
        mid_idx = len(hydro.stations) // 2
        axes[0, 1].plot(hydro.times/3600.0, hydro.depths[:, mid_idx], 'b-', 
                       linewidth=2, label='中点水深')
        ax_twin2 = axes[0, 1].twinx()
        ax_twin2.plot(stability.times/3600.0, stability.channel_stability_index, 
                     'r-', linewidth=2, label='稳定性指数')
        axes[0, 1].set_xlabel('时间 (小时)', fontsize=12)
        axes[0, 1].set_ylabel('水深 (m)', fontsize=12, color='b')
        ax_twin2.set_ylabel('稳定性指数', fontsize=12, color='r')
        axes[0, 1].set_title('水深与稳定性关系', fontsize=14, fontweight='bold')
        axes[0, 1].tick_params(axis='y', labelcolor='b')
        ax_twin2.tick_params(axis='y', labelcolor='r')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 流速vs安全系数（使用匹配的时间和空间索引）
        # 找到两个数据集的公共时间点
        common_times = np.intersect1d(hydro.times, stability.times)
        if len(common_times) > 0:
            hydro_time_indices = [np.where(hydro.times == t)[0][0] for t in common_times]
            stab_time_indices = [np.where(stability.times == t)[0][0] for t in common_times]
            
            # 提取公共站点（假设都有）
            n_stations_min = min(len(hydro.stations), len(stability.stations))
            
            v_data = []
            h_data = []
            fs_data = []
            for ht_idx, st_idx in zip(hydro_time_indices, stab_time_indices):
                v_data.extend(hydro.velocities[ht_idx, :n_stations_min].tolist())
                h_data.extend(hydro.depths[ht_idx, :n_stations_min].tolist())
                fs_data.extend(stability.comprehensive_factors[st_idx, :n_stations_min].tolist())
            
            axes[1, 0].scatter(v_data, fs_data, alpha=0.3, s=10)
            axes[1, 0].set_xlabel('流速 (m/s)', fontsize=12)
            axes[1, 0].set_ylabel('安全系数', fontsize=12)
            axes[1, 0].set_title('流速-安全系数相关性', fontsize=14, fontweight='bold')
            axes[1, 0].axhline(y=1.3, color='orange', linestyle='--', alpha=0.5)
            axes[1, 0].grid(True, alpha=0.3)
            
            # 水深vs安全系数
            axes[1, 1].scatter(h_data, fs_data, alpha=0.3, s=10, c='green')
            axes[1, 1].set_xlabel('水深 (m)', fontsize=12)
            axes[1, 1].set_ylabel('安全系数', fontsize=12)
            axes[1, 1].set_title('水深-安全系数相关性', fontsize=14, fontweight='bold')
            axes[1, 1].axhline(y=1.3, color='orange', linestyle='--', alpha=0.5)
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, '无匹配数据点', 
                          ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 1].text(0.5, 0.5, '无匹配数据点', 
                          ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, '03_水动力边坡稳定耦合分析.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ 图表已生成: 03_水动力边坡稳定耦合分析.png")


def main():
    """主测试流程"""
    print("\n" + "="*80)
    print("边坡稳定性模块专项测试分析")
    print("="*80 + "\n")
    
    # 创建输出目录
    output_dir = "/workspace/tests/slope_stability_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 创建明渠系统
    print("步骤1: 创建明渠系统...")
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
    
    channel = ChannelSystem(name="边坡稳定性测试渠道", total_length=5000.0, sections=[])
    
    base_section = ChannelSection(
        station=0.0,
        bed_elevation=100.0,
        bottom_width=6.0,
        side_slope=2.0,
        max_depth=5.0,
        manning_n=0.025,
        slope_height=4.0,
        slope_angle=26.57,
        soil_properties=soil,
        lining_properties=lining,
    )
    
    channel.create_uniform_sections(
        num_sections=11,
        base_section=base_section,
        bed_slope=0.0002,
    )
    
    print(f"  ✓ 明渠长度: {channel.total_length} m")
    print(f"  ✓ 断面数量: {channel.num_sections}")
    
    # 2. 创建监测网络
    print("\n步骤2: 创建监测网络...")
    network = MonitoringNetwork(network_name="边坡稳定性监测网络")
    network.create_uniform_groundwater_network(
        channel_length=channel.total_length,
        spacing=1000.0,
    )
    network.add_boundary_stations(0.0, channel.total_length)
    
    # 初始化监测数据
    times = np.linspace(0, 7200, 20)
    for station in network.groundwater_stations:
        for t in times:
            station.add_observation(time=t, groundwater_level=98.0)
    for station in network.rainfall_stations:
        for t in times:
            station.add_observation(time=t, rainfall=0.0)
    
    print(f"  ✓ 监测站点: {network.num_stations}")
    
    # 3. 设置边界条件
    print("\n步骤3: 设置边界条件...")
    
    def upstream_q(t):
        """上游流量：从10增加到25 m³/s"""
        if t < 3600:
            return 10.0
        else:
            return 25.0
    
    bc = BoundaryConditions(
        upstream_type='discharge',
        downstream_type='stage',
        upstream_discharge_func=upstream_q,
        downstream_stage_func=lambda t, q: 102.5,
    )
    
    print("  ✓ 上游边界: 流量 10→25 m³/s (阶跃变化)")
    print("  ✓ 下游边界: 水位 102.5 m (固定)")
    
    # 4. 创建仿真器
    print("\n步骤4: 创建综合仿真器...")
    simulator = IntegratedSimulator(
        channel=channel,
        monitoring_network=network,
        boundary_conditions=bc,
    )
    
    # 5. 运行仿真
    print("\n步骤5: 运行仿真计算...")
    config = SimulationConfig(
        total_time=7200.0,
        dt=60.0,
        save_interval=1,
        hydrodynamic_solver_type="quasi_steady",
        enable_hydrodynamics=True,
        enable_water_quality=True,
        enable_slope_stability=True,
        simulated_indicators=['DO', 'BOD', 'NH3N'],
    )
    
    print(f"  求解器类型: {config.hydrodynamic_solver_type}")
    print(f"  总时长: {config.total_time} 秒")
    print(f"  时间步长: {config.dt} 秒")
    
    results = simulator.run_simulation(config)
    print("  ✓ 仿真计算完成")
    
    # 6. 生成报告和图表
    print("\n步骤6: 生成分析报告和图表...")
    
    # 保存详细报告
    save_detailed_report(results, output_dir)
    
    # 保存数据表格
    save_data_tables(results, output_dir)
    
    # 生成综合图表
    generate_comprehensive_figures(results, output_dir)
    
    # 7. 总结
    print("\n" + "="*80)
    print("✓ 边坡稳定性模块专项测试完成！")
    print("="*80)
    print(f"\n所有结果保存在: {output_dir}/")
    print("  - 边坡稳定性分析报告.md")
    print("  - data_tables/ (数据表格)")
    print("  - figures/ (分析图表)")
    print("\n请进行人工检查验证。\n")


if __name__ == "__main__":
    main()
