"""
生成边坡稳定性分析的动画结果
展示纵剖面图的时间演化过程
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.patches as patches
from datetime import datetime

from channel_stability.core.channel_system import (
    ChannelSystem, ChannelSection, SoilProperties, LiningProperties
)
from channel_stability.core.monitoring_network import MonitoringNetwork
from channel_stability.hydrodynamics.boundary_conditions import BoundaryConditions
from channel_stability.integrated_simulation import IntegratedSimulator, SimulationConfig


def create_longitudinal_profile_animation(results, output_dir, filename="longitudinal_profile.gif"):
    """
    创建纵剖面图动画
    
    Parameters
    ----------
    results : IntegratedResults
        仿真结果
    output_dir : str
        输出目录
    filename : str
        输出文件名
    """
    hydro = results.hydrodynamics
    stability = results.slope_stability
    
    if hydro is None:
        print("  ⚠ 无水动力学数据，跳过纵剖面动画")
        return
    
    print(f"\n生成纵剖面图动画: {filename}")
    
    # 创建图形
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :])  # 水面线
    ax2 = fig.add_subplot(gs[1, 0])  # 流速分布
    ax3 = fig.add_subplot(gs[1, 1])  # 流量分布
    ax4 = fig.add_subplot(gs[2, 0])  # 安全系数分布
    ax5 = fig.add_subplot(gs[2, 1])  # Froude数分布
    
    # 获取数据范围
    x = hydro.stations / 1000.0  # 转换为km
    bed_elevation = 100.0 - 0.0002 * hydro.stations  # 渠底高程
    
    times_h = hydro.times / 3600.0
    n_frames = len(hydro.times)
    
    # 初始化图形元素
    line_water, = ax1.plot([], [], 'b-', linewidth=2, label='水面线')
    line_bed, = ax1.plot(x, bed_elevation, 'k-', linewidth=2, label='渠底')
    
    line_velocity, = ax2.plot([], [], 'g-', linewidth=2)
    line_discharge, = ax3.plot([], [], 'purple', linewidth=2)
    line_safety, = ax4.plot([], [], 'r-', linewidth=2)
    line_froude, = ax5.plot([], [], 'orange', linewidth=2)
    
    time_text = fig.text(0.5, 0.95, '', ha='center', fontsize=14, fontweight='bold')
    
    # 设置坐标轴
    ax1.set_xlabel('沿程位置 (km)', fontsize=12)
    ax1.set_ylabel('高程 (m)', fontsize=12)
    ax1.set_title('纵剖面 - 水面线演化', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, x[-1])
    y_min = bed_elevation.min() - 1
    y_max = bed_elevation.max() + np.max(hydro.depths) + 1
    ax1.set_ylim(y_min, y_max)
    
    ax2.set_xlabel('沿程位置 (km)', fontsize=12)
    ax2.set_ylabel('流速 (m/s)', fontsize=12)
    ax2.set_title('流速分布', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, x[-1])
    ax2.set_ylim(0, np.max(hydro.velocities) * 1.2)
    ax2.axhline(y=5.0, color='red', linestyle='--', alpha=0.3, label='高速阈值')
    
    ax3.set_xlabel('沿程位置 (km)', fontsize=12)
    ax3.set_ylabel('流量 (m³/s)', fontsize=12)
    ax3.set_title('流量分布', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, x[-1])
    ax3.set_ylim(0, np.max(hydro.discharges) * 1.2)
    
    if stability is not None:
        ax4.set_xlabel('沿程位置 (km)', fontsize=12)
        ax4.set_ylabel('安全系数', fontsize=12)
        ax4.set_title('边坡安全系数分布', fontsize=12, fontweight='bold')
        ax4.axhline(y=1.3, color='orange', linestyle='--', alpha=0.5, label='安全阈值')
        ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='临界')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, x[-1])
        ax4.set_ylim(0, np.max(stability.comprehensive_factors) * 1.2)
        ax4.legend()
    
    ax5.set_xlabel('沿程位置 (km)', fontsize=12)
    ax5.set_ylabel('Froude数', fontsize=12)
    ax5.set_title('Froude数分布', fontsize=12, fontweight='bold')
    ax5.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='临界Fr=1')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, x[-1])
    ax5.set_ylim(0, max(1.5, np.max(hydro.froude_numbers) * 1.2))
    ax5.legend()
    
    def init():
        """初始化动画"""
        line_water.set_data([], [])
        line_velocity.set_data([], [])
        line_discharge.set_data([], [])
        line_safety.set_data([], [])
        line_froude.set_data([], [])
        time_text.set_text('')
        return line_water, line_velocity, line_discharge, line_safety, line_froude, time_text
    
    def update(frame):
        """更新动画帧"""
        # 更新水面线
        water_surface = bed_elevation + hydro.depths[frame, :]
        line_water.set_data(x, water_surface)
        
        # 更新流速
        line_velocity.set_data(x, hydro.velocities[frame, :])
        
        # 更新流量
        line_discharge.set_data(x, hydro.discharges[frame, :])
        
        # 更新Froude数
        line_froude.set_data(x, hydro.froude_numbers[frame, :])
        
        # 更新安全系数（如果有）
        if stability is not None and frame < len(stability.times):
            # 找到最接近的稳定性时间步
            time_diff = np.abs(stability.times - hydro.times[frame])
            stab_idx = np.argmin(time_diff)
            x_stab = stability.stations / 1000.0
            line_safety.set_data(x_stab, stability.comprehensive_factors[stab_idx, :])
        
        # 更新时间文本
        time_text.set_text(f'时间: {times_h[frame]:.2f} 小时 | ' + 
                          f'Q={hydro.discharges[frame, 0]:.1f} m³/s | ' +
                          f'V_max={np.max(hydro.velocities[frame, :]):.2f} m/s')
        
        return line_water, line_velocity, line_discharge, line_safety, line_froude, time_text
    
    # 创建动画
    print(f"  正在生成 {n_frames} 帧动画...")
    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                        blit=True, interval=100, repeat=True)
    
    # 保存为GIF
    output_file = os.path.join(output_dir, filename)
    writer = PillowWriter(fps=10)
    anim.save(output_file, writer=writer, dpi=100)
    plt.close()
    
    print(f"  ✓ 动画已保存: {output_file}")


def create_water_quality_animation(results, output_dir, filename="water_quality.gif"):
    """
    创建水质传播过程动画
    
    Parameters
    ----------
    results : IntegratedResults
        仿真结果
    output_dir : str
        输出目录
    filename : str
        输出文件名
    """
    wq = results.water_quality
    
    if wq is None:
        print("  ⚠ 无水质数据，跳过水质动画")
        return
    
    print(f"\n生成水质传播动画: {filename}")
    
    # 获取指标列表
    indicators = list(wq.concentrations.keys())
    n_indicators = len(indicators)
    
    if n_indicators == 0:
        print("  ⚠ 无水质指标数据")
        return
    
    # 创建图形
    fig, axes = plt.subplots(n_indicators, 1, figsize=(14, 4*n_indicators))
    if n_indicators == 1:
        axes = [axes]
    
    x = wq.stations / 1000.0  # 转换为km
    times_h = wq.times / 3600.0
    n_frames = len(wq.times)
    
    # 初始化每个指标的图形元素
    lines = []
    fill_areas = []
    
    for idx, (indicator, ax) in enumerate(zip(indicators, axes)):
        conc = wq.concentrations[indicator]
        
        line, = ax.plot([], [], 'b-', linewidth=2, label=f'{indicator}浓度')
        lines.append(line)
        
        # 添加填充区域
        fill = ax.fill_between(x, 0, 0, alpha=0.3)
        fill_areas.append(fill)
        
        ax.set_xlabel('沿程位置 (km)', fontsize=12)
        ax.set_ylabel(f'{indicator} 浓度 (mg/L)', fontsize=12)
        ax.set_title(f'{indicator} 浓度纵向分布', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, x[-1])
        ax.set_ylim(np.min(conc) * 0.9, np.max(conc) * 1.1)
        ax.legend()
    
    time_text = fig.text(0.5, 0.98, '', ha='center', fontsize=14, fontweight='bold')
    
    def init():
        """初始化动画"""
        for line in lines:
            line.set_data([], [])
        time_text.set_text('')
        return tuple(lines) + (time_text,)
    
    def update(frame):
        """更新动画帧"""
        for idx, indicator in enumerate(indicators):
            conc = wq.concentrations[indicator][frame, :]
            lines[idx].set_data(x, conc)
            
            # 更新填充区域
            for coll in axes[idx].collections:
                coll.remove()
            axes[idx].fill_between(x, 0, conc, alpha=0.3, color='blue')
        
        # 更新时间文本
        time_text.set_text(f'时间: {times_h[frame]:.2f} 小时')
        
        return tuple(lines) + (time_text,)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # 创建动画
    print(f"  正在生成 {n_frames} 帧动画...")
    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                        blit=False, interval=100, repeat=True)
    
    # 保存为GIF
    output_file = os.path.join(output_dir, filename)
    writer = PillowWriter(fps=10)
    anim.save(output_file, writer=writer, dpi=100)
    plt.close()
    
    print(f"  ✓ 动画已保存: {output_file}")


def create_cross_section_animation(results, output_dir, filename="cross_section.gif"):
    """
    创建横断面动画（展示水位、边坡稳定性）
    
    Parameters
    ----------
    results : IntegratedResults
        仿真结果
    output_dir : str
        输出目录
    filename : str
        输出文件名
    """
    hydro = results.hydrodynamics
    stability = results.slope_stability
    
    if hydro is None:
        print("  ⚠ 无水动力学数据，跳过横断面动画")
        return
    
    print(f"\n生成横断面动画: {filename}")
    
    # 选择中间断面
    mid_idx = len(hydro.stations) // 2
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    times_h = hydro.times / 3600.0
    n_frames = len(hydro.times)
    
    # 渠道几何参数（从配置获取，这里使用示例值）
    bottom_width = 6.0
    side_slope = 2.0  # 边坡坡度 (H:V)
    slope_height = 4.0
    
    def draw_channel_section(ax, water_depth, safety_factor=None):
        """绘制渠道横断面"""
        ax.clear()
        
        # 绘制渠道断面
        x_left = -bottom_width/2 - side_slope * slope_height
        x_right = bottom_width/2 + side_slope * slope_height
        
        # 渠道轮廓
        channel_x = [x_left, -bottom_width/2, bottom_width/2, x_right]
        channel_y = [slope_height, 0, 0, slope_height]
        ax.plot(channel_x, channel_y, 'k-', linewidth=3, label='渠道边界')
        ax.fill(channel_x, channel_y, color='wheat', alpha=0.3)
        
        # 绘制水面
        if water_depth > 0:
            water_width_top = bottom_width + 2 * side_slope * water_depth
            water_x = [-water_width_top/2, -bottom_width/2, 
                      bottom_width/2, water_width_top/2]
            water_y = [water_depth, 0, 0, water_depth]
            ax.fill(water_x, water_y, color='cyan', alpha=0.5, label='水体')
            ax.plot([-water_width_top/2, water_width_top/2], 
                   [water_depth, water_depth], 'b-', linewidth=2, label='水面')
        
        # 绘制边坡稳定性状态
        if safety_factor is not None:
            if safety_factor >= 1.5:
                color, status = 'green', '非常安全'
            elif safety_factor >= 1.3:
                color, status = 'yellow', '安全'
            elif safety_factor >= 1.0:
                color, status = 'orange', '基本稳定'
            else:
                color, status = 'red', '不稳定'
            
            # 在边坡上标注
            ax.text(x_left + 1, slope_height * 0.8, 
                   f'左坡\nFS={safety_factor:.2f}\n{status}',
                   ha='center', fontsize=10, color=color, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            ax.text(x_right - 1, slope_height * 0.8, 
                   f'右坡\nFS={safety_factor:.2f}\n{status}',
                   ha='center', fontsize=10, color=color, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        ax.set_xlim(x_left - 2, x_right + 2)
        ax.set_ylim(-0.5, slope_height + 1)
        ax.set_xlabel('横向距离 (m)', fontsize=12)
        ax.set_ylabel('高程 (m)', fontsize=12)
        ax.set_title(f'横断面 (桩号 {hydro.stations[mid_idx]:.0f}m)', 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    # 初始化第二个子图（水深时间序列）
    line_depth, = ax2.plot([], [], 'b-', linewidth=2, label='水深')
    ax2.set_xlabel('时间 (小时)', fontsize=12)
    ax2.set_ylabel('水深 (m)', fontsize=12)
    ax2.set_title('中间断面水深演化', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, times_h[-1])
    ax2.set_ylim(0, np.max(hydro.depths[:, mid_idx]) * 1.2)
    ax2.axhline(y=slope_height, color='red', linestyle='--', 
               alpha=0.5, label='渠道最大深度')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    time_marker, = ax2.plot([], [], 'ro', markersize=10)
    
    time_text = fig.text(0.5, 0.95, '', ha='center', fontsize=14, fontweight='bold')
    
    def update(frame):
        """更新动画帧"""
        water_depth = hydro.depths[frame, mid_idx]
        
        # 获取安全系数
        fs = None
        if stability is not None and frame < len(stability.times):
            time_diff = np.abs(stability.times - hydro.times[frame])
            stab_idx = np.argmin(time_diff)
            fs = stability.comprehensive_factors[stab_idx, mid_idx]
        
        # 绘制横断面
        draw_channel_section(ax1, water_depth, fs)
        
        # 更新水深时间序列
        line_depth.set_data(times_h[:frame+1], hydro.depths[:frame+1, mid_idx])
        time_marker.set_data([times_h[frame]], [water_depth])
        
        # 更新时间文本
        time_text.set_text(f'时间: {times_h[frame]:.2f} 小时 | 水深: {water_depth:.2f} m | ' +
                          f'流量: {hydro.discharges[frame, mid_idx]:.1f} m³/s')
        
        return time_text,
    
    # 创建动画
    print(f"  正在生成 {n_frames} 帧动画...")
    anim = FuncAnimation(fig, update, frames=n_frames,
                        blit=False, interval=100, repeat=True)
    
    # 保存为GIF
    output_file = os.path.join(output_dir, filename)
    writer = PillowWriter(fps=10)
    anim.save(output_file, writer=writer, dpi=100)
    plt.close()
    
    print(f"  ✓ 动画已保存: {output_file}")


def create_comprehensive_animation(results, output_dir, filename="comprehensive.gif"):
    """
    创建综合动画（所有参数在一个图中）
    
    Parameters
    ----------
    results : IntegratedResults
        仿真结果
    output_dir : str
        输出目录
    filename : str
        输出文件名
    """
    hydro = results.hydrodynamics
    stability = results.slope_stability
    wq = results.water_quality
    
    if hydro is None:
        print("  ⚠ 无水动力学数据，跳过综合动画")
        return
    
    print(f"\n生成综合动画: {filename}")
    
    # 创建图形
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.25)
    
    ax1 = fig.add_subplot(gs[0, :])  # 水面线
    ax2 = fig.add_subplot(gs[1, 0])  # 流速
    ax3 = fig.add_subplot(gs[1, 1])  # 流量
    ax4 = fig.add_subplot(gs[2, 0])  # 安全系数
    ax5 = fig.add_subplot(gs[2, 1])  # Froude数
    ax6 = fig.add_subplot(gs[3, :])  # 水质（如果有）
    
    x = hydro.stations / 1000.0
    bed_elevation = 100.0 - 0.0002 * hydro.stations
    times_h = hydro.times / 3600.0
    n_frames = len(hydro.times)
    
    # 初始化图形元素
    lines = {}
    
    # 水面线
    lines['bed'], = ax1.plot(x, bed_elevation, 'k-', linewidth=2, label='渠底')
    lines['water'], = ax1.plot([], [], 'b-', linewidth=2.5, label='水面线')
    ax1.fill_between(x, bed_elevation, bed_elevation, alpha=0.2, color='blue')
    ax1.set_xlabel('沿程位置 (km)', fontsize=11)
    ax1.set_ylabel('高程 (m)', fontsize=11)
    ax1.set_title('纵剖面图', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, x[-1])
    ax1.set_ylim(bed_elevation.min() - 1, bed_elevation.max() + np.max(hydro.depths) + 1)
    
    # 流速
    lines['velocity'], = ax2.plot([], [], 'g-', linewidth=2)
    ax2.axhline(y=5.0, color='red', linestyle='--', alpha=0.3)
    ax2.set_xlabel('沿程位置 (km)', fontsize=11)
    ax2.set_ylabel('流速 (m/s)', fontsize=11)
    ax2.set_title('流速分布', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, x[-1])
    ax2.set_ylim(0, np.max(hydro.velocities) * 1.2)
    
    # 流量
    lines['discharge'], = ax3.plot([], [], 'purple', linewidth=2)
    ax3.set_xlabel('沿程位置 (km)', fontsize=11)
    ax3.set_ylabel('流量 (m³/s)', fontsize=11)
    ax3.set_title('流量分布', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, x[-1])
    ax3.set_ylim(0, np.max(hydro.discharges) * 1.2)
    
    # 安全系数
    if stability is not None:
        x_stab = stability.stations / 1000.0
        lines['safety'], = ax4.plot([], [], 'r-', linewidth=2)
        ax4.axhline(y=1.3, color='orange', linestyle='--', alpha=0.5)
        ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
        ax4.set_xlabel('沿程位置 (km)', fontsize=11)
        ax4.set_ylabel('安全系数', fontsize=11)
        ax4.set_title('边坡安全系数', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, x[-1])
        ax4.set_ylim(0, np.max(stability.comprehensive_factors) * 1.2)
    else:
        ax4.text(0.5, 0.5, '无边坡稳定性数据', ha='center', va='center',
                transform=ax4.transAxes, fontsize=12)
    
    # Froude数
    lines['froude'], = ax5.plot([], [], 'orange', linewidth=2)
    ax5.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    ax5.set_xlabel('沿程位置 (km)', fontsize=11)
    ax5.set_ylabel('Froude数', fontsize=11)
    ax5.set_title('Froude数分布', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, x[-1])
    ax5.set_ylim(0, max(1.5, np.max(hydro.froude_numbers) * 1.2))
    
    # 水质
    if wq is not None and len(wq.concentrations) > 0:
        # 选择第一个指标
        indicator = list(wq.concentrations.keys())[0]
        x_wq = wq.stations / 1000.0
        lines['wq'], = ax6.plot([], [], 'b-', linewidth=2, label=f'{indicator}')
        ax6.set_xlabel('沿程位置 (km)', fontsize=11)
        ax6.set_ylabel(f'{indicator} 浓度 (mg/L)', fontsize=11)
        ax6.set_title(f'{indicator} 浓度分布', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.set_xlim(0, x[-1])
        conc = wq.concentrations[indicator]
        ax6.set_ylim(np.min(conc) * 0.9, np.max(conc) * 1.1)
        ax6.legend()
    else:
        ax6.text(0.5, 0.5, '无水质数据', ha='center', va='center',
                transform=ax6.transAxes, fontsize=12)
    
    time_text = fig.text(0.5, 0.97, '', ha='center', fontsize=15, fontweight='bold')
    
    def update(frame):
        """更新动画帧"""
        # 水面线
        water_surface = bed_elevation + hydro.depths[frame, :]
        lines['water'].set_data(x, water_surface)
        for coll in ax1.collections:
            coll.remove()
        ax1.fill_between(x, bed_elevation, water_surface, alpha=0.3, color='cyan')
        
        # 流速
        lines['velocity'].set_data(x, hydro.velocities[frame, :])
        
        # 流量
        lines['discharge'].set_data(x, hydro.discharges[frame, :])
        
        # Froude数
        lines['froude'].set_data(x, hydro.froude_numbers[frame, :])
        
        # 安全系数
        if stability is not None and 'safety' in lines:
            time_diff = np.abs(stability.times - hydro.times[frame])
            stab_idx = np.argmin(time_diff)
            lines['safety'].set_data(x_stab, stability.comprehensive_factors[stab_idx, :])
        
        # 水质
        if wq is not None and 'wq' in lines and frame < len(wq.times):
            indicator = list(wq.concentrations.keys())[0]
            lines['wq'].set_data(x_wq, wq.concentrations[indicator][frame, :])
        
        # 时间文本
        time_text.set_text(
            f'时间: {times_h[frame]:.2f} 小时 | '
            f'流量: {hydro.discharges[frame, 0]:.1f} m³/s | '
            f'最大流速: {np.max(hydro.velocities[frame, :]):.2f} m/s | '
            f'最大Fr: {np.max(hydro.froude_numbers[frame, :]):.3f}'
        )
        
        return tuple(lines.values()) + (time_text,)
    
    # 创建动画
    print(f"  正在生成 {n_frames} 帧动画...")
    anim = FuncAnimation(fig, update, frames=n_frames,
                        blit=False, interval=100, repeat=True)
    
    # 保存为GIF
    output_file = os.path.join(output_dir, filename)
    writer = PillowWriter(fps=10)
    anim.save(output_file, writer=writer, dpi=120)
    plt.close()
    
    print(f"  ✓ 动画已保存: {output_file}")


def main():
    """主程序"""
    print("\n" + "="*80)
    print("生成边坡稳定性分析动画")
    print("="*80 + "\n")
    
    # 创建输出目录
    output_dir = "/workspace/tests/slope_stability_analysis/animations"
    os.makedirs(output_dir, exist_ok=True)
    
    print("步骤1: 创建明渠系统并运行仿真...")
    
    # 创建系统（与之前相同）
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
    
    channel = ChannelSystem(name="动画示例渠道", total_length=5000.0, sections=[])
    
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
    
    # 创建监测网络
    network = MonitoringNetwork(network_name="动画监测网络")
    network.create_uniform_groundwater_network(
        channel_length=channel.total_length,
        spacing=1000.0,
    )
    network.add_boundary_stations(0.0, channel.total_length)
    
    times = np.linspace(0, 7200, 20)
    for station in network.groundwater_stations:
        for t in times:
            station.add_observation(time=t, groundwater_level=98.0)
    for station in network.rainfall_stations:
        for t in times:
            station.add_observation(time=t, rainfall=0.0)
    
    # 边界条件
    def upstream_q(t):
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
    
    # 创建仿真器并运行
    simulator = IntegratedSimulator(
        channel=channel,
        monitoring_network=network,
        boundary_conditions=bc,
    )
    
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
    
    print("  正在运行仿真...")
    results = simulator.run_simulation(config)
    print("  ✓ 仿真完成\n")
    
    # 生成各种动画
    print("步骤2: 生成动画文件...")
    
    # 1. 纵剖面综合动画
    create_longitudinal_profile_animation(results, output_dir, "01_纵剖面综合.gif")
    
    # 2. 水质传播动画
    create_water_quality_animation(results, output_dir, "02_水质传播.gif")
    
    # 3. 横断面动画
    create_cross_section_animation(results, output_dir, "03_横断面演化.gif")
    
    # 4. 综合动画
    create_comprehensive_animation(results, output_dir, "04_综合展示.gif")
    
    print("\n" + "="*80)
    print("✓ 所有动画生成完成！")
    print("="*80)
    print(f"\n动画文件保存在: {output_dir}/")
    print("  - 01_纵剖面综合.gif")
    print("  - 02_水质传播.gif")
    print("  - 03_横断面演化.gif")
    print("  - 04_综合展示.gif")
    print("\n请查看并进行人工检查。\n")


if __name__ == "__main__":
    main()
