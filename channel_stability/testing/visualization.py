"""
动态可视化工具

生成时间序列图表和沿程分布的GIF动画
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Tuple
import os


class DynamicVisualizer:
    """动态可视化器"""
    
    def __init__(self, output_dir: str = './results'):
        """
        初始化可视化器
        
        Parameters
        ----------
        output_dir : str
            输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_timeseries(
        self,
        results: Dict,
        station_indices: List[int],
        output_prefix: str = 'timeseries',
        scenario_info: Optional[Dict] = None,
    ) -> List[str]:
        """
        绘制时间序列图
        
        Parameters
        ----------
        results : Dict
            仿真结果字典，包含 hydrodynamics, water_quality, slope_stability
        station_indices : List[int]
            需要绘制的断面索引列表
        output_prefix : str
            输出文件名前缀
        scenario_info : Optional[Dict]
            场景信息（用于显示边界条件变化）
        
        Returns
        -------
        output_files : List[str]
            生成的文件路径列表
        """
        output_files = []
        
        hydro = results.get('hydrodynamics')
        wq = results.get('water_quality')
        stability = results.get('slope_stability')
        
        for idx in station_indices:
            fig = plt.figure(figsize=(16, 12))
            gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)
            
            station = hydro.stations[idx] if hydro else 0
            fig.suptitle(f'断面 {station:.0f}m 处的时空动态特征', fontsize=16, fontweight='bold')
            
            # 1. 水深时间序列
            if hydro is not None:
                ax1 = fig.add_subplot(gs[0, 0])
                times_h = hydro.times / 3600.0  # 转换为小时
                depths = hydro.depths[:, idx]
                ax1.plot(times_h, depths, 'b-', linewidth=2, label='水深')
                ax1.set_xlabel('时间 (小时)')
                ax1.set_ylabel('水深 (m)')
                ax1.set_title('水深变化')
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                
                # 添加阶跃时刻标记
                if scenario_info and 'step_time' in scenario_info:
                    step_t = scenario_info['step_time'] / 3600.0
                    ax1.axvline(step_t, color='r', linestyle='--', alpha=0.5, label='阶跃时刻')
            
            # 2. 流量时间序列
            if hydro is not None:
                ax2 = fig.add_subplot(gs[0, 1])
                discharges = hydro.discharges[:, idx]
                ax2.plot(times_h, discharges, 'g-', linewidth=2, label='流量')
                ax2.set_xlabel('时间 (小时)')
                ax2.set_ylabel('流量 (m³/s)')
                ax2.set_title('流量变化')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                
                if scenario_info and 'step_time' in scenario_info:
                    step_t = scenario_info['step_time'] / 3600.0
                    ax2.axvline(step_t, color='r', linestyle='--', alpha=0.5)
            
            # 3. 流速和Froude数
            if hydro is not None:
                ax3 = fig.add_subplot(gs[1, 0])
                velocities = hydro.velocities[:, idx]
                ax3.plot(times_h, velocities, 'm-', linewidth=2, label='流速')
                ax3.set_xlabel('时间 (小时)')
                ax3.set_ylabel('流速 (m/s)')
                ax3.set_title('流速变化')
                ax3.grid(True, alpha=0.3)
                ax3.legend()
                
                ax3b = ax3.twinx()
                froude = hydro.froude_numbers[:, idx]
                ax3b.plot(times_h, froude, 'c--', linewidth=2, label='Froude数')
                ax3b.set_ylabel('Froude数')
                ax3b.legend(loc='upper right')
            
            # 4. 水质指标
            if wq is not None:
                ax4 = fig.add_subplot(gs[1, 1])
                times_wq = wq.times / 3600.0
                
                colors = ['orange', 'brown', 'purple', 'pink', 'olive']
                for i, (indicator, conc) in enumerate(wq.concentrations.items()):
                    color = colors[i % len(colors)]
                    ax4.plot(times_wq, conc[:, idx], '-', linewidth=2, 
                            color=color, label=indicator)
                
                ax4.set_xlabel('时间 (小时)')
                ax4.set_ylabel('浓度 (mg/L)')
                ax4.set_title('水质指标变化')
                ax4.grid(True, alpha=0.3)
                ax4.legend()
                
                if scenario_info and 'step_time' in scenario_info:
                    step_t = scenario_info['step_time'] / 3600.0
                    ax4.axvline(step_t, color='r', linestyle='--', alpha=0.5)
            
            # 5. 边坡稳定性系数
            if stability is not None:
                ax5 = fig.add_subplot(gs[2, 0])
                times_s = stability.times / 3600.0
                
                # 找到最接近的断面索引
                station_idx = self._find_nearest_index(stability.stations, station)
                
                factors = stability.comprehensive_factors[:, station_idx]
                ax5.plot(times_s, factors, 'r-', linewidth=2, label='综合安全系数')
                ax5.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='临界值')
                ax5.set_xlabel('时间 (小时)')
                ax5.set_ylabel('安全系数')
                ax5.set_title('边坡综合安全系数')
                ax5.grid(True, alpha=0.3)
                ax5.legend()
            
            # 6. 各稳定性分量
            if stability is not None:
                ax6 = fig.add_subplot(gs[2, 1])
                
                sliding = stability.sliding_factors[:, station_idx]
                overturning = stability.overturning_factors[:, station_idx]
                uplift = stability.uplift_factors[:, station_idx]
                seepage = stability.seepage_factors[:, station_idx]
                
                ax6.plot(times_s, sliding, '-', linewidth=1.5, label='抗滑稳定')
                ax6.plot(times_s, overturning, '-', linewidth=1.5, label='抗倾覆稳定')
                ax6.plot(times_s, uplift, '-', linewidth=1.5, label='抗浮稳定')
                ax6.plot(times_s, seepage, '-', linewidth=1.5, label='渗透稳定')
                ax6.axhline(1.0, color='k', linestyle='--', alpha=0.5)
                
                ax6.set_xlabel('时间 (小时)')
                ax6.set_ylabel('安全系数')
                ax6.set_title('各稳定性分量')
                ax6.grid(True, alpha=0.3)
                ax6.legend()
            
            # 7. 场景信息文本
            ax7 = fig.add_subplot(gs[3, :])
            ax7.axis('off')
            
            info_text = f"断面位置: {station:.0f} m\n"
            if scenario_info:
                info_text += f"场景: {scenario_info.get('name', 'N/A')}\n"
                info_text += f"描述: {scenario_info.get('description', 'N/A')}\n"
                if 'baseline_value' in scenario_info:
                    info_text += f"基准值: {scenario_info['baseline_value']:.2f}\n"
                if 'step_value' in scenario_info:
                    info_text += f"阶跃值: {scenario_info['step_value']:.2f}\n"
                if 'step_time' in scenario_info:
                    info_text += f"阶跃时刻: {scenario_info['step_time']/3600:.1f} 小时\n"
            
            ax7.text(0.05, 0.5, info_text, fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # 保存
            output_file = os.path.join(
                self.output_dir, 
                f'{output_prefix}_station_{int(station)}.png'
            )
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            output_files.append(output_file)
            print(f"已生成时间序列图: {output_file}")
        
        return output_files
    
    def create_spatial_animation(
        self,
        results: Dict,
        variable: str = 'depth',
        output_filename: str = 'spatial_animation.gif',
        scenario_info: Optional[Dict] = None,
        fps: int = 10,
    ) -> str:
        """
        创建沿程分布的GIF动画
        
        Parameters
        ----------
        results : Dict
            仿真结果
        variable : str
            要展示的变量: 'depth', 'discharge', 'velocity', 'water_quality', 'stability'
        output_filename : str
            输出文件名
        scenario_info : Optional[Dict]
            场景信息
        fps : int
            帧率
        
        Returns
        -------
        output_file : str
            输出文件路径
        """
        hydro = results.get('hydrodynamics')
        wq = results.get('water_quality')
        stability = results.get('slope_stability')
        
        if hydro is None:
            raise ValueError("需要水动力学结果")
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        times = hydro.times
        stations = hydro.stations
        
        def animate(frame_idx):
            for ax in axes:
                ax.clear()
            
            t = times[frame_idx]
            t_hour = t / 3600.0
            
            # 上图：主要变量沿程分布
            ax1 = axes[0]
            
            if variable == 'depth':
                depths = hydro.depths[frame_idx, :]
                water_levels = hydro.water_levels[frame_idx, :]
                bed_elevs = water_levels - depths
                
                ax1.fill_between(stations, bed_elevs, water_levels, 
                                alpha=0.5, color='blue', label='水体')
                ax1.plot(stations, water_levels, 'b-', linewidth=2, label='水面线')
                ax1.plot(stations, bed_elevs, 'k-', linewidth=1.5, label='渠底')
                ax1.set_ylabel('高程 (m)')
                ax1.set_title(f'水深沿程分布 - 时间: {t_hour:.2f} 小时')
                ax1.legend(loc='upper right')
                ax1.grid(True, alpha=0.3)
                
            elif variable == 'discharge':
                discharges = hydro.discharges[frame_idx, :]
                ax1.plot(stations, discharges, 'g-', linewidth=2, marker='o', markersize=4)
                ax1.set_ylabel('流量 (m³/s)')
                ax1.set_title(f'流量沿程分布 - 时间: {t_hour:.2f} 小时')
                ax1.grid(True, alpha=0.3)
                
            elif variable == 'velocity':
                velocities = hydro.velocities[frame_idx, :]
                froude = hydro.froude_numbers[frame_idx, :]
                
                ax1.plot(stations, velocities, 'm-', linewidth=2, marker='o', 
                        markersize=4, label='流速')
                ax1.set_ylabel('流速 (m/s)', color='m')
                ax1.tick_params(axis='y', labelcolor='m')
                
                ax1b = ax1.twinx()
                ax1b.plot(stations, froude, 'c--', linewidth=2, marker='s', 
                         markersize=4, label='Froude数')
                ax1b.set_ylabel('Froude数', color='c')
                ax1b.tick_params(axis='y', labelcolor='c')
                
                ax1.set_title(f'流速和Froude数沿程分布 - 时间: {t_hour:.2f} 小时')
                ax1.grid(True, alpha=0.3)
            
            # 下图：水质或稳定性
            ax2 = axes[1]
            
            if wq is not None and variable in ['water_quality', 'depth', 'discharge', 'velocity']:
                # 显示水质指标
                for indicator, conc in wq.concentrations.items():
                    # 找到最近的时间索引
                    wq_idx = self._find_nearest_index(wq.times, t)
                    ax2.plot(wq.stations, conc[wq_idx, :], '-o', linewidth=2, 
                            markersize=4, label=indicator)
                
                ax2.set_xlabel('桩号 (m)')
                ax2.set_ylabel('浓度 (mg/L)')
                ax2.set_title('水质指标沿程分布')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            elif stability is not None:
                # 显示稳定性系数
                stab_idx = self._find_nearest_index(stability.times, t)
                
                factors = stability.comprehensive_factors[stab_idx, :]
                ax2.plot(stability.stations, factors, 'r-', linewidth=2, 
                        marker='o', markersize=4, label='综合安全系数')
                ax2.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='临界值')
                ax2.fill_between(stability.stations, 0, factors, 
                                where=(factors < 1.0), alpha=0.3, color='red')
                
                ax2.set_xlabel('桩号 (m)')
                ax2.set_ylabel('安全系数')
                ax2.set_title('边坡稳定性沿程分布')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim([0, max(3.0, np.max(factors) * 1.1)])
            
            # 添加场景信息
            if scenario_info:
                info_text = f"{scenario_info.get('name', '')}"
                if 'step_time' in scenario_info:
                    if t >= scenario_info['step_time']:
                        info_text += f" [已阶跃]"
                    else:
                        info_text += f" [阶跃前]"
                
                fig.text(0.5, 0.95, info_text, ha='center', fontsize=12, 
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
            
            return axes
        
        # 创建动画
        num_frames = len(times)
        # 采样以减少文件大小
        sample_interval = max(1, num_frames // 100)  # 最多100帧
        frame_indices = list(range(0, num_frames, sample_interval))
        
        anim = animation.FuncAnimation(
            fig, animate, frames=frame_indices,
            interval=1000//fps, blit=False, repeat=True
        )
        
        output_file = os.path.join(self.output_dir, output_filename)
        anim.save(output_file, writer='pillow', fps=fps, dpi=100)
        plt.close()
        
        print(f"已生成动画: {output_file}")
        return output_file
    
    def plot_boundary_conditions(
        self,
        scenario: Dict,
        total_time: float,
        output_filename: str = 'boundary_conditions.png',
    ) -> str:
        """
        绘制边界条件变化图
        
        Parameters
        ----------
        scenario : Dict
            场景对象
        total_time : float
            总时间 (s)
        output_filename : str
            输出文件名
        
        Returns
        -------
        output_file : str
            输出文件路径
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        times = np.linspace(0, total_time, 1000)
        times_h = times / 3600.0
        
        if 'time_function' in scenario:
            values = [scenario['time_function'](t) for t in times]
            
            ax.plot(times_h, values, 'b-', linewidth=2)
            
            if 'step_time' in scenario:
                step_t = scenario['step_time'] / 3600.0
                ax.axvline(step_t, color='r', linestyle='--', linewidth=2, 
                          label=f'阶跃时刻: {step_t:.1f}h')
                
                baseline = scenario.get('baseline_value', 0)
                step_val = scenario.get('step_value', 0)
                ax.axhline(baseline, color='g', linestyle=':', alpha=0.5, 
                          label=f'基准值: {baseline:.2f}')
                ax.axhline(step_val, color='orange', linestyle=':', alpha=0.5, 
                          label=f'阶跃值: {step_val:.2f}')
        
        ax.set_xlabel('时间 (小时)')
        ax.set_ylabel(scenario.get('parameter_type', '值'))
        ax.set_title(f"边界条件: {scenario.get('name', '')}\n{scenario.get('description', '')}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        output_file = os.path.join(self.output_dir, output_filename)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"已生成边界条件图: {output_file}")
        return output_file
    
    def create_comparison_plot(
        self,
        results_list: List[Dict],
        scenario_names: List[str],
        variable: str = 'depth',
        station_index: int = 5,
        output_filename: str = 'comparison.png',
    ) -> str:
        """
        创建多场景对比图
        
        Parameters
        ----------
        results_list : List[Dict]
            多个结果字典列表
        scenario_names : List[str]
            场景名称列表
        variable : str
            对比的变量
        station_index : int
            断面索引
        output_filename : str
            输出文件名
        
        Returns
        -------
        output_file : str
            输出文件路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(results_list)))
        
        for i, (results, name) in enumerate(zip(results_list, scenario_names)):
            hydro = results.get('hydrodynamics')
            if hydro is None:
                continue
            
            times_h = hydro.times / 3600.0
            station = hydro.stations[station_index]
            color = colors[i]
            
            # 水深
            axes[0].plot(times_h, hydro.depths[:, station_index], 
                        linewidth=2, label=name, color=color)
            
            # 流量
            axes[1].plot(times_h, hydro.discharges[:, station_index], 
                        linewidth=2, label=name, color=color)
            
            # 流速
            axes[2].plot(times_h, hydro.velocities[:, station_index], 
                        linewidth=2, label=name, color=color)
            
            # Froude数
            axes[3].plot(times_h, hydro.froude_numbers[:, station_index], 
                        linewidth=2, label=name, color=color)
        
        axes[0].set_xlabel('时间 (小时)')
        axes[0].set_ylabel('水深 (m)')
        axes[0].set_title('水深对比')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        axes[1].set_xlabel('时间 (小时)')
        axes[1].set_ylabel('流量 (m³/s)')
        axes[1].set_title('流量对比')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        axes[2].set_xlabel('时间 (小时)')
        axes[2].set_ylabel('流速 (m/s)')
        axes[2].set_title('流速对比')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        axes[3].set_xlabel('时间 (小时)')
        axes[3].set_ylabel('Froude数')
        axes[3].set_title('Froude数对比')
        axes[3].grid(True, alpha=0.3)
        axes[3].legend()
        
        fig.suptitle(f'多场景对比分析 - 断面 {station:.0f}m', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_file = os.path.join(self.output_dir, output_filename)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"已生成对比图: {output_file}")
        return output_file
    
    @staticmethod
    def _find_nearest_index(array: np.ndarray, value: float) -> int:
        """找到数组中最接近给定值的索引"""
        return int(np.argmin(np.abs(array - value)))
