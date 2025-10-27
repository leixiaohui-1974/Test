"""
计算结果质量评价程序

评价内容：
1. 稳态流计算收敛性
2. 非恒定流数值稳定性
3. 边坡稳定性计算正确性
4. 物理合理性检查
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class QualityMetrics:
    """质量指标"""
    name: str
    passed: bool
    value: float
    threshold: float
    message: str


@dataclass
class EvaluationResults:
    """评价结果"""
    scenario_name: str
    steady_flow_metrics: List[QualityMetrics]
    unsteady_flow_metrics: List[QualityMetrics]
    stability_metrics: List[QualityMetrics]
    physical_metrics: List[QualityMetrics]
    overall_score: float
    overall_passed: bool
    detailed_diagnostics: Dict


class ComputationQualityEvaluator:
    """计算质量评价器"""
    
    # 物理合理性阈值
    PHYSICAL_THRESHOLDS = {
        'max_velocity': 5.0,  # m/s，明渠最大合理流速
        'max_froude': 1.5,  # 明渠通常为缓流
        'max_depth': 6.0,  # m
        'min_depth': 0.01,  # m
        'min_safety_factor': 1.0,  # 边坡安全系数最小值
        'max_safety_factor': 10.0,  # 边坡安全系数最大值
    }
    
    # 数值稳定性阈值
    NUMERICAL_THRESHOLDS = {
        'max_velocity_change_rate': 0.5,  # 相邻时间步最大流速变化率
        'max_depth_change_rate': 0.3,  # 相邻时间步最大水深变化率
        'mass_conservation_error': 0.01,  # 质量守恒误差 (1%)
        'min_courant_number': 0.1,
        'max_courant_number': 2.0,
    }
    
    def __init__(self, output_dir: str = './tests/evaluation_results'):
        """初始化评价器"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def evaluate_scenario(
        self,
        scenario_name: str,
        results_file: str,
        scenario_info: Optional[Dict] = None,
    ) -> EvaluationResults:
        """
        评价单个场景的计算结果
        
        Parameters
        ----------
        scenario_name : str
            场景名称
        results_file : str
            结果文件路径 (.npz)
        scenario_info : Optional[Dict]
            场景信息
        
        Returns
        -------
        evaluation : EvaluationResults
            评价结果
        """
        print(f"\n{'='*70}")
        print(f"评价场景: {scenario_name}")
        print(f"{'='*70}")
        
        # 加载结果
        data = np.load(results_file)
        
        # 提取数据
        hydro_times = data['hydro_times']
        hydro_stations = data['hydro_stations']
        depths = data['depths']
        discharges = data['discharges']
        velocities = data['velocities']
        water_levels = data['water_levels']
        froude_numbers = data['froude_numbers']
        
        stability_times = data['stability_times']
        stability_stations = data['stability_stations']
        comprehensive_factors = data['comprehensive_factors']
        sliding_factors = data['sliding_factors']
        overturning_factors = data['overturning_factors']
        uplift_factors = data['uplift_factors']
        seepage_factors = data['seepage_factors']
        
        # 评价各方面
        steady_metrics = self._evaluate_steady_flow(
            depths, discharges, velocities, hydro_stations
        )
        
        unsteady_metrics = self._evaluate_unsteady_flow(
            hydro_times, depths, discharges, velocities, 
            froude_numbers, hydro_stations
        )
        
        stability_metrics = self._evaluate_stability(
            stability_times, comprehensive_factors, sliding_factors,
            overturning_factors, uplift_factors, seepage_factors
        )
        
        physical_metrics = self._evaluate_physical_reasonability(
            depths, velocities, froude_numbers, comprehensive_factors
        )
        
        # 计算总分
        all_metrics = (
            steady_metrics + unsteady_metrics + 
            stability_metrics + physical_metrics
        )
        
        passed_count = sum(1 for m in all_metrics if m.passed)
        total_count = len(all_metrics)
        overall_score = passed_count / total_count if total_count > 0 else 0
        overall_passed = overall_score >= 0.8  # 80%通过率
        
        # 详细诊断
        diagnostics = self._generate_diagnostics(
            hydro_times, depths, discharges, velocities,
            froude_numbers, comprehensive_factors,
            hydro_stations, stability_stations
        )
        
        results = EvaluationResults(
            scenario_name=scenario_name,
            steady_flow_metrics=steady_metrics,
            unsteady_flow_metrics=unsteady_metrics,
            stability_metrics=stability_metrics,
            physical_metrics=physical_metrics,
            overall_score=overall_score,
            overall_passed=overall_passed,
            detailed_diagnostics=diagnostics,
        )
        
        # 生成报告和图表
        self._generate_report(results, scenario_name)
        self._generate_plots(
            scenario_name, hydro_times, hydro_stations,
            depths, discharges, velocities, froude_numbers,
            comprehensive_factors, stability_times, stability_stations,
            diagnostics
        )
        
        return results
    
    def _evaluate_steady_flow(
        self,
        depths: np.ndarray,
        discharges: np.ndarray,
        velocities: np.ndarray,
        stations: np.ndarray,
    ) -> List[QualityMetrics]:
        """评价稳态流计算"""
        metrics = []
        
        # 初始状态（假设第一个时间步是稳态）
        h0 = depths[0, :]
        q0 = discharges[0, :]
        v0 = velocities[0, :]
        
        # 1. 流量连续性
        q_mean = np.mean(q0)
        q_std = np.std(q0)
        q_variation = q_std / q_mean if q_mean > 0 else np.inf
        
        metrics.append(QualityMetrics(
            name="流量连续性",
            passed=q_variation < 0.05,  # 5%变化
            value=q_variation,
            threshold=0.05,
            message=f"沿程流量变化: {q_variation*100:.2f}% (应<5%)"
        ))
        
        # 2. 水深合理性
        h_min, h_max = np.min(h0), np.max(h0)
        h_reasonable = (h_min > 0.01) and (h_max < 6.0)
        
        metrics.append(QualityMetrics(
            name="水深范围",
            passed=h_reasonable,
            value=h_max,
            threshold=6.0,
            message=f"水深范围: [{h_min:.3f}, {h_max:.3f}] m"
        ))
        
        # 3. 流速合理性
        v_max = np.max(v0)
        v_reasonable = v_max < 5.0
        
        metrics.append(QualityMetrics(
            name="初始流速",
            passed=v_reasonable,
            value=v_max,
            threshold=5.0,
            message=f"最大流速: {v_max:.3f} m/s (应<5 m/s)"
        ))
        
        return metrics
    
    def _evaluate_unsteady_flow(
        self,
        times: np.ndarray,
        depths: np.ndarray,
        discharges: np.ndarray,
        velocities: np.ndarray,
        froude_numbers: np.ndarray,
        stations: np.ndarray,
    ) -> List[QualityMetrics]:
        """评价非恒定流数值稳定性"""
        metrics = []
        
        nt, nx = depths.shape
        dt = times[1] - times[0] if len(times) > 1 else 1.0
        dx = stations[1] - stations[0] if len(stations) > 1 else 1.0
        
        # 1. 时间步变化率
        if nt > 1:
            depth_changes = np.abs(np.diff(depths, axis=0)) / dt
            max_depth_change_rate = np.max(depth_changes)
            
            metrics.append(QualityMetrics(
                name="水深时间变化率",
                passed=max_depth_change_rate < 0.1,  # m/s
                value=max_depth_change_rate,
                threshold=0.1,
                message=f"最大水深变化率: {max_depth_change_rate:.4f} m/s"
            ))
            
            velocity_changes = np.abs(np.diff(velocities, axis=0)) / dt
            max_velocity_change_rate = np.max(velocity_changes)
            
            metrics.append(QualityMetrics(
                name="流速时间变化率",
                passed=max_velocity_change_rate < 1.0,  # m/s²
                value=max_velocity_change_rate,
                threshold=1.0,
                message=f"最大流速变化率: {max_velocity_change_rate:.4f} m/s²"
            ))
        
        # 2. Courant数
        courant = np.abs(velocities) * dt / dx
        courant_max = np.max(courant)
        courant_mean = np.mean(courant)
        
        metrics.append(QualityMetrics(
            name="Courant数范围",
            passed=(courant_max < 2.0) and (courant_mean < 1.0),
            value=courant_max,
            threshold=2.0,
            message=f"Courant数: 最大={courant_max:.3f}, 平均={courant_mean:.3f}"
        ))
        
        # 3. 流速物理合理性
        v_max = np.max(velocities)
        v_min = np.min(velocities)
        
        metrics.append(QualityMetrics(
            name="流速物理范围",
            passed=(v_max < 5.0) and (v_min >= 0),
            value=v_max,
            threshold=5.0,
            message=f"流速范围: [{v_min:.3f}, {v_max:.3f}] m/s (应在[0, 5])"
        ))
        
        # 4. Froude数合理性
        fr_max = np.max(froude_numbers)
        fr_min = np.min(froude_numbers)
        
        metrics.append(QualityMetrics(
            name="Froude数范围",
            passed=(fr_max < 1.5) and (fr_min >= 0),
            value=fr_max,
            threshold=1.5,
            message=f"Froude数范围: [{fr_min:.3f}, {fr_max:.3f}] (缓流应<1)"
        ))
        
        # 5. 质量守恒
        if nx > 1:
            q_inlet = discharges[:, 0]
            q_outlet = discharges[:, -1]
            q_mean = (q_inlet + q_outlet) / 2
            q_diff = np.abs(q_outlet - q_inlet)
            mass_error = np.mean(q_diff / (q_mean + 1e-10))
            
            metrics.append(QualityMetrics(
                name="质量守恒",
                passed=mass_error < 0.05,  # 5%
                value=mass_error,
                threshold=0.05,
                message=f"质量守恒误差: {mass_error*100:.2f}% (应<5%)"
            ))
        
        # 6. 数值振荡检测
        if nt > 2 and nx > 2:
            # 检测水深的高频振荡
            depth_grad_t = np.diff(depths, axis=0)
            oscillation = np.sum(np.abs(np.diff(depth_grad_t, axis=0)))
            oscillation_normalized = oscillation / (nt * nx * np.mean(depths))
            
            metrics.append(QualityMetrics(
                name="数值振荡",
                passed=oscillation_normalized < 0.1,
                value=oscillation_normalized,
                threshold=0.1,
                message=f"振荡指标: {oscillation_normalized:.4f} (应<0.1)"
            ))
        
        return metrics
    
    def _evaluate_stability(
        self,
        times: np.ndarray,
        comprehensive: np.ndarray,
        sliding: np.ndarray,
        overturning: np.ndarray,
        uplift: np.ndarray,
        seepage: np.ndarray,
    ) -> List[QualityMetrics]:
        """评价边坡稳定性计算"""
        metrics = []
        
        # 1. 安全系数范围
        fs_min = np.min(comprehensive)
        fs_max = np.max(comprehensive)
        
        metrics.append(QualityMetrics(
            name="综合安全系数范围",
            passed=(fs_min > 0) and (fs_max < 20),
            value=fs_max,
            threshold=20.0,
            message=f"安全系数范围: [{fs_min:.3f}, {fs_max:.3f}]"
        ))
        
        # 2. 各分项系数合理性
        for name, factors in [
            ('滑动', sliding),
            ('倾覆', overturning),
            ('浮托', uplift),
            ('渗透', seepage)
        ]:
            f_min = np.min(factors)
            f_max = np.max(factors)
            f_reasonable = (f_min > 0) and (f_max < 50)
            
            metrics.append(QualityMetrics(
                name=f"{name}安全系数",
                passed=f_reasonable,
                value=f_max,
                threshold=50.0,
                message=f"{name}系数: [{f_min:.3f}, {f_max:.3f}]"
            ))
        
        # 3. 时间变化合理性
        if len(times) > 1:
            fs_changes = np.abs(np.diff(comprehensive, axis=0))
            max_change = np.max(fs_changes)
            
            metrics.append(QualityMetrics(
                name="安全系数时间变化",
                passed=max_change < 2.0,
                value=max_change,
                threshold=2.0,
                message=f"最大单步变化: {max_change:.3f}"
            ))
        
        # 4. 检测常数值问题
        # 如果所有值都相同，说明计算有问题
        fs_std = np.std(comprehensive)
        fs_mean = np.mean(comprehensive)
        variation = fs_std / fs_mean if fs_mean > 0 else 0
        
        metrics.append(QualityMetrics(
            name="安全系数变异性",
            passed=variation > 0.01,  # 至少有1%变化
            value=variation,
            threshold=0.01,
            message=f"变异系数: {variation:.4f} (常数值问题)" if variation < 0.01 
                    else f"变异系数: {variation:.4f}"
        ))
        
        return metrics
    
    def _evaluate_physical_reasonability(
        self,
        depths: np.ndarray,
        velocities: np.ndarray,
        froude_numbers: np.ndarray,
        safety_factors: np.ndarray,
    ) -> List[QualityMetrics]:
        """评价物理合理性"""
        metrics = []
        
        # 统计所有物理量
        stats = {
            '水深': (np.min(depths), np.max(depths), np.mean(depths)),
            '流速': (np.min(velocities), np.max(velocities), np.mean(velocities)),
            'Froude数': (np.min(froude_numbers), np.max(froude_numbers), np.mean(froude_numbers)),
            '安全系数': (np.min(safety_factors), np.max(safety_factors), np.mean(safety_factors)),
        }
        
        # 物理量范围检查
        checks = [
            ('水深正值', np.all(depths > 0), "所有水深应为正"),
            ('流速正值', np.all(velocities >= 0), "所有流速应非负"),
            ('Froude数正值', np.all(froude_numbers >= 0), "所有Froude数应非负"),
            ('安全系数正值', np.all(safety_factors > 0), "所有安全系数应为正"),
            ('无NaN值', not np.any(np.isnan(depths)) and not np.any(np.isnan(velocities)), 
             "不应有NaN值"),
            ('无Inf值', not np.any(np.isinf(velocities)) and not np.any(np.isinf(safety_factors)), 
             "不应有Inf值"),
        ]
        
        for name, passed, message in checks:
            metrics.append(QualityMetrics(
                name=name,
                passed=passed,
                value=1.0 if passed else 0.0,
                threshold=1.0,
                message=message
            ))
        
        return metrics
    
    def _generate_diagnostics(
        self,
        times: np.ndarray,
        depths: np.ndarray,
        discharges: np.ndarray,
        velocities: np.ndarray,
        froude_numbers: np.ndarray,
        safety_factors: np.ndarray,
        hydro_stations: np.ndarray,
        stability_stations: np.ndarray,
    ) -> Dict:
        """生成详细诊断信息"""
        
        diagnostics = {
            '模拟时间': {
                '总时长(s)': float(times[-1] - times[0]) if len(times) > 1 else 0,
                '时间步数': len(times),
                '平均时间步长(s)': float(np.mean(np.diff(times))) if len(times) > 1 else 0,
            },
            '空间离散': {
                '断面数': len(hydro_stations),
                '总长度(m)': float(hydro_stations[-1] - hydro_stations[0]),
                '平均空间步长(m)': float(np.mean(np.diff(hydro_stations))) if len(hydro_stations) > 1 else 0,
            },
            '水深统计': {
                '最小值(m)': float(np.min(depths)),
                '最大值(m)': float(np.max(depths)),
                '平均值(m)': float(np.mean(depths)),
                '标准差(m)': float(np.std(depths)),
            },
            '流速统计': {
                '最小值(m/s)': float(np.min(velocities)),
                '最大值(m/s)': float(np.max(velocities)),
                '平均值(m/s)': float(np.mean(velocities)),
                '标准差(m/s)': float(np.std(velocities)),
                '异常高值数量': int(np.sum(velocities > 5.0)),
            },
            '流量统计': {
                '最小值(m³/s)': float(np.min(discharges)),
                '最大值(m³/s)': float(np.max(discharges)),
                '平均值(m³/s)': float(np.mean(discharges)),
                '标准差(m³/s)': float(np.std(discharges)),
            },
            'Froude数统计': {
                '最小值': float(np.min(froude_numbers)),
                '最大值': float(np.max(froude_numbers)),
                '平均值': float(np.mean(froude_numbers)),
                '超临界流比例': float(np.sum(froude_numbers > 1.0) / froude_numbers.size),
                '异常高值数量': int(np.sum(froude_numbers > 2.0)),
            },
            '安全系数统计': {
                '最小值': float(np.min(safety_factors)),
                '最大值': float(np.max(safety_factors)),
                '平均值': float(np.mean(safety_factors)),
                '不稳定比例': float(np.sum(safety_factors < 1.0) / safety_factors.size),
            },
        }
        
        # 识别问题时间步和位置
        problem_times = []
        problem_locations = []
        
        # 找到异常流速的位置
        abnormal_v = np.where(velocities > 10.0)
        if len(abnormal_v[0]) > 0:
            for t_idx, x_idx in zip(abnormal_v[0][:5], abnormal_v[1][:5]):  # 最多显示5个
                problem_times.append({
                    '时间(s)': float(times[t_idx]),
                    '位置(m)': float(hydro_stations[x_idx]),
                    '流速(m/s)': float(velocities[t_idx, x_idx]),
                    '问题': '流速异常过大'
                })
        
        # 找到异常Froude数的位置
        abnormal_fr = np.where(froude_numbers > 2.0)
        if len(abnormal_fr[0]) > 0:
            for t_idx, x_idx in zip(abnormal_fr[0][:5], abnormal_fr[1][:5]):
                problem_times.append({
                    '时间(s)': float(times[t_idx]),
                    '位置(m)': float(hydro_stations[x_idx]),
                    'Froude数': float(froude_numbers[t_idx, x_idx]),
                    '问题': 'Froude数异常过大'
                })
        
        diagnostics['异常点识别'] = problem_times[:10]  # 最多显示10个
        
        return diagnostics
    
    def _generate_report(
        self,
        results: EvaluationResults,
        scenario_name: str,
    ):
        """生成评价报告"""
        
        report_file = os.path.join(self.output_dir, f'{scenario_name}_评价报告.md')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# 计算结果质量评价报告\n\n")
            f.write(f"**场景**: {scenario_name}\n\n")
            f.write(f"**评价时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**总体得分**: {results.overall_score*100:.1f}% ")
            f.write(f"({'✓ 通过' if results.overall_passed else '✗ 不通过'})\n\n")
            
            f.write("---\n\n")
            
            # 各部分评价
            sections = [
                ('稳态流计算', results.steady_flow_metrics),
                ('非恒定流数值稳定性', results.unsteady_flow_metrics),
                ('边坡稳定性计算', results.stability_metrics),
                ('物理合理性', results.physical_metrics),
            ]
            
            for section_name, metrics in sections:
                f.write(f"## {section_name}\n\n")
                
                passed = sum(1 for m in metrics if m.passed)
                total = len(metrics)
                f.write(f"**通过率**: {passed}/{total} ({passed/total*100:.1f}%)\n\n")
                
                f.write("| 检查项 | 状态 | 数值 | 阈值 | 说明 |\n")
                f.write("|--------|------|------|------|------|\n")
                
                for m in metrics:
                    status = "✓" if m.passed else "✗"
                    f.write(f"| {m.name} | {status} | {m.value:.4f} | {m.threshold:.4f} | {m.message} |\n")
                
                f.write("\n")
            
            # 详细诊断
            f.write("## 详细诊断信息\n\n")
            
            diag = results.detailed_diagnostics
            
            for section, data in diag.items():
                if section == '异常点识别':
                    continue
                    
                f.write(f"### {section}\n\n")
                for key, value in data.items():
                    if isinstance(value, float):
                        f.write(f"- **{key}**: {value:.4f}\n")
                    else:
                        f.write(f"- **{key}**: {value}\n")
                f.write("\n")
            
            # 异常点
            if '异常点识别' in diag and diag['异常点识别']:
                f.write("### 异常点识别\n\n")
                f.write("发现以下异常点:\n\n")
                for i, point in enumerate(diag['异常点识别'], 1):
                    f.write(f"{i}. ")
                    for key, value in point.items():
                        if isinstance(value, float):
                            f.write(f"{key}={value:.3f}, ")
                        else:
                            f.write(f"{key}={value}, ")
                    f.write("\n")
                f.write("\n")
            
            # 结论和建议
            f.write("## 结论和建议\n\n")
            
            if results.overall_passed:
                f.write("✓ **总体结论**: 计算结果质量良好，通过评价。\n\n")
            else:
                f.write("✗ **总体结论**: 计算结果存在质量问题，需要改进。\n\n")
                
                f.write("### 主要问题\n\n")
                
                failed_metrics = []
                for metrics in [results.steady_flow_metrics, results.unsteady_flow_metrics,
                               results.stability_metrics, results.physical_metrics]:
                    failed_metrics.extend([m for m in metrics if not m.passed])
                
                for m in failed_metrics:
                    f.write(f"- **{m.name}**: {m.message}\n")
                
                f.write("\n### 改进建议\n\n")
                
                # 根据失败的指标给出建议
                if any('流速' in m.name or 'Froude' in m.name for m in failed_metrics):
                    f.write("1. **数值稳定性问题**:\n")
                    f.write("   - 减小时间步长，确保满足CFL条件\n")
                    f.write("   - 检查边界条件设置是否合理\n")
                    f.write("   - 考虑使用更稳定的数值格式或添加人工粘性\n\n")
                
                if any('质量守恒' in m.name for m in failed_metrics):
                    f.write("2. **质量守恒问题**:\n")
                    f.write("   - 检查边界条件实施是否正确\n")
                    f.write("   - 验证数值格式的守恒性\n")
                    f.write("   - 减小时间和空间步长\n\n")
                
                if any('安全系数' in m.name or '稳定性' in m.name for m in failed_metrics):
                    f.write("3. **边坡稳定性计算问题**:\n")
                    f.write("   - 检查土壤参数和衬砌参数设置\n")
                    f.write("   - 验证地下水位和降雨数据的插值方法\n")
                    f.write("   - 检查各失稳模式的计算公式\n\n")
        
        print(f"✓ 评价报告已生成: {report_file}")
    
    def _generate_plots(
        self,
        scenario_name: str,
        times: np.ndarray,
        stations: np.ndarray,
        depths: np.ndarray,
        discharges: np.ndarray,
        velocities: np.ndarray,
        froude_numbers: np.ndarray,
        safety_factors: np.ndarray,
        stability_times: np.ndarray,
        stability_stations: np.ndarray,
        diagnostics: Dict,
    ):
        """生成诊断图表"""
        
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        fig.suptitle(f'计算结果质量诊断 - {scenario_name}', 
                     fontsize=16, fontweight='bold')
        
        times_h = times / 3600.0
        
        # 选择中间断面
        mid_idx = len(stations) // 2
        
        # 1. 水深时间历程
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(times_h, depths[:, mid_idx], 'b-', linewidth=2)
        ax1.set_xlabel('时间 (小时)')
        ax1.set_ylabel('水深 (m)')
        ax1.set_title(f'水深时间历程 (桩号{stations[mid_idx]:.0f}m)')
        ax1.grid(True, alpha=0.3)
        
        # 2. 流速时间历程
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(times_h, velocities[:, mid_idx], 'g-', linewidth=2)
        ax2.axhline(y=5.0, color='r', linestyle='--', label='合理上限')
        ax2.set_xlabel('时间 (小时)')
        ax2.set_ylabel('流速 (m/s)')
        ax2.set_title(f'流速时间历程 (桩号{stations[mid_idx]:.0f}m)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Froude数时间历程
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(times_h, froude_numbers[:, mid_idx], 'm-', linewidth=2)
        ax3.axhline(y=1.0, color='r', linestyle='--', label='临界流')
        ax3.axhline(y=1.5, color='orange', linestyle='--', label='合理上限')
        ax3.set_xlabel('时间 (小时)')
        ax3.set_ylabel('Froude数')
        ax3.set_title(f'Froude数时间历程 (桩号{stations[mid_idx]:.0f}m)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 流量沿程分布（最后时刻）
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(stations, discharges[-1, :], 'b-o', linewidth=2, markersize=4)
        ax4.set_xlabel('桩号 (m)')
        ax4.set_ylabel('流量 (m³/s)')
        ax4.set_title('流量沿程分布 (最后时刻)')
        ax4.grid(True, alpha=0.3)
        
        # 5. 流速沿程分布（最后时刻）
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(stations, velocities[-1, :], 'g-o', linewidth=2, markersize=4)
        ax5.axhline(y=5.0, color='r', linestyle='--', alpha=0.5)
        ax5.set_xlabel('桩号 (m)')
        ax5.set_ylabel('流速 (m/s)')
        ax5.set_title('流速沿程分布 (最后时刻)')
        ax5.grid(True, alpha=0.3)
        
        # 6. Courant数分布
        ax6 = fig.add_subplot(gs[1, 2])
        if len(times) > 1 and len(stations) > 1:
            dt = times[1] - times[0]
            dx = stations[1] - stations[0]
            courant = np.abs(velocities) * dt / dx
            im = ax6.imshow(courant.T, aspect='auto', origin='lower',
                           extent=[times_h[0], times_h[-1], stations[0], stations[-1]],
                           cmap='RdYlGn_r', vmin=0, vmax=2)
            plt.colorbar(im, ax=ax6, label='Courant数')
            ax6.set_xlabel('时间 (小时)')
            ax6.set_ylabel('桩号 (m)')
            ax6.set_title('Courant数时空分布')
        
        # 7. 安全系数时间历程
        ax7 = fig.add_subplot(gs[2, 0])
        mid_stab_idx = len(stability_stations) // 2
        stab_times_h = stability_times / 3600.0
        ax7.plot(stab_times_h, safety_factors[:, mid_stab_idx], 'r-', linewidth=2)
        ax7.axhline(y=1.0, color='k', linestyle='--', label='临界值')
        ax7.axhline(y=1.3, color='orange', linestyle='--', label='安全阈值')
        ax7.set_xlabel('时间 (小时)')
        ax7.set_ylabel('综合安全系数')
        ax7.set_title(f'安全系数时间历程 (桩号{stability_stations[mid_stab_idx]:.0f}m)')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. 安全系数沿程分布
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.plot(stability_stations, safety_factors[-1, :], 'r-o', linewidth=2, markersize=4)
        ax8.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
        ax8.axhline(y=1.3, color='orange', linestyle='--', alpha=0.5)
        ax8.fill_between(stability_stations, 0, safety_factors[-1, :],
                        where=(safety_factors[-1, :] < 1.0), alpha=0.3, color='red')
        ax8.set_xlabel('桩号 (m)')
        ax8.set_ylabel('综合安全系数')
        ax8.set_title('安全系数沿程分布 (最后时刻)')
        ax8.grid(True, alpha=0.3)
        
        # 9. 统计信息文本
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        
        text_lines = [
            "关键统计指标:",
            "",
            f"水深: [{np.min(depths):.3f}, {np.max(depths):.3f}] m",
            f"流速: [{np.min(velocities):.3f}, {np.max(velocities):.3f}] m/s",
            f"Froude: [{np.min(froude_numbers):.3f}, {np.max(froude_numbers):.3f}]",
            f"安全系数: [{np.min(safety_factors):.3f}, {np.max(safety_factors):.3f}]",
            "",
            f"异常流速(>5m/s): {np.sum(velocities > 5.0)} 个",
            f"异常Froude(>2): {np.sum(froude_numbers > 2.0)} 个",
            f"不稳定断面(<1): {np.sum(safety_factors < 1.0)} 个",
            "",
            f"模拟时长: {times[-1]/3600:.2f} 小时",
            f"时间步数: {len(times)}",
            f"断面数: {len(stations)}",
        ]
        
        text = "\n".join(text_lines)
        ax9.text(0.1, 0.5, text, fontsize=11, verticalalignment='center',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 保存
        output_file = os.path.join(self.output_dir, f'{scenario_name}_诊断图.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 诊断图已生成: {output_file}")


def main():
    """主函数"""
    print("\n" + "="*80)
    print("计算结果质量评价程序")
    print("="*80 + "\n")
    
    evaluator = ComputationQualityEvaluator()
    
    # 查找所有场景结果
    test_dir = './tests/dynamic_scenarios_20251027_013510'
    
    if not os.path.exists(test_dir):
        print(f"错误: 测试目录不存在: {test_dir}")
        return
    
    scenarios = []
    for item in os.listdir(test_dir):
        scenario_dir = os.path.join(test_dir, item)
        if os.path.isdir(scenario_dir):
            results_file = os.path.join(scenario_dir, 'results.npz')
            if os.path.exists(results_file):
                scenarios.append((item, results_file))
    
    print(f"找到 {len(scenarios)} 个场景待评价\n")
    
    # 评价每个场景
    all_results = []
    for scenario_name, results_file in scenarios:
        try:
            result = evaluator.evaluate_scenario(
                scenario_name=scenario_name,
                results_file=results_file,
            )
            all_results.append(result)
        except Exception as e:
            print(f"\n✗ 评价场景 '{scenario_name}' 失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 生成汇总报告
    print("\n" + "="*80)
    print("生成汇总报告...")
    print("="*80 + "\n")
    
    summary_file = os.path.join(evaluator.output_dir, '汇总评价报告.md')
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("# 计算结果质量评价汇总报告\n\n")
        f.write(f"**评价时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**评价场景数**: {len(all_results)}\n\n")
        
        f.write("## 场景评价汇总\n\n")
        f.write("| 场景名称 | 总体得分 | 稳态流 | 非恒定流 | 稳定性 | 物理性 | 状态 |\n")
        f.write("|---------|---------|--------|---------|--------|--------|------|\n")
        
        for result in all_results:
            steady_score = sum(1 for m in result.steady_flow_metrics if m.passed) / len(result.steady_flow_metrics)
            unsteady_score = sum(1 for m in result.unsteady_flow_metrics if m.passed) / len(result.unsteady_flow_metrics)
            stability_score = sum(1 for m in result.stability_metrics if m.passed) / len(result.stability_metrics)
            physical_score = sum(1 for m in result.physical_metrics if m.passed) / len(result.physical_metrics)
            
            status = "✓" if result.overall_passed else "✗"
            
            f.write(f"| {result.scenario_name} | {result.overall_score*100:.1f}% | "
                   f"{steady_score*100:.1f}% | {unsteady_score*100:.1f}% | "
                   f"{stability_score*100:.1f}% | {physical_score*100:.1f}% | {status} |\n")
        
        f.write("\n## 主要发现\n\n")
        
        # 统计常见问题
        all_failed = []
        for result in all_results:
            for metrics in [result.steady_flow_metrics, result.unsteady_flow_metrics,
                           result.stability_metrics, result.physical_metrics]:
                all_failed.extend([m.name for m in metrics if not m.passed])
        
        from collections import Counter
        problem_counts = Counter(all_failed)
        
        f.write("### 最常见的问题 (按出现频率排序)\n\n")
        for problem, count in problem_counts.most_common(10):
            f.write(f"{count}. **{problem}**: 在 {count}/{len(all_results)} 个场景中出现\n")
        
        f.write("\n### 总体建议\n\n")
        f.write("1. **数值稳定性**: 需要检查时间步长设置和CFL条件\n")
        f.write("2. **边界条件**: 建议验证边界条件的物理合理性和数值实现\n")
        f.write("3. **初值计算**: 稳态流初值计算可能需要改进收敛判据\n")
        f.write("4. **边坡稳定性**: 检查参数设置和计算公式，确保正确响应水力条件变化\n")
        
        f.write("\n---\n\n")
        f.write(f"*详细评价报告请查看各场景的单独报告*\n")
    
    print(f"\n✓ 汇总报告已生成: {summary_file}")
    
    print("\n" + "="*80)
    print("评价完成！")
    print(f"结果保存在: {evaluator.output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
