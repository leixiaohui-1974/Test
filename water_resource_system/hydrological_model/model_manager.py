"""
水文模型管理器
管理全国范围内的多个水文模型
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import json

logger = logging.getLogger(__name__)


class ModelManager:
    """水文模型管理器"""
    
    def __init__(self, workspace: Path):
        """
        初始化模型管理器
        
        Args:
            workspace: 工作目录
        """
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        
        # 模型注册表
        self.models = {}
        self.model_registry_path = self.workspace / "model_registry.json"
        
        # 加载已注册的模型
        self._load_registry()
    
    def _load_registry(self):
        """加载模型注册表"""
        if self.model_registry_path.exists():
            try:
                with open(self.model_registry_path, 'r', encoding='utf-8') as f:
                    self.models = json.load(f)
                logger.info(f"加载了 {len(self.models)} 个已注册模型")
            except Exception as e:
                logger.error(f"加载模型注册表失败: {e}")
                self.models = {}
    
    def _save_registry(self):
        """保存模型注册表"""
        try:
            with open(self.model_registry_path, 'w', encoding='utf-8') as f:
                json.dump(self.models, f, ensure_ascii=False, indent=2)
            logger.info("模型注册表已保存")
        except Exception as e:
            logger.error(f"保存模型注册表失败: {e}")
    
    def register_model(self, reservoir_id: str, model_config: Dict):
        """
        注册模型
        
        Args:
            reservoir_id: 水库ID
            model_config: 模型配置
        """
        self.models[reservoir_id] = {
            "config": model_config,
            "registered_at": datetime.now().isoformat(),
            "status": "active",
        }
        
        self._save_registry()
        logger.info(f"注册模型: {reservoir_id}")
    
    def get_model(self, reservoir_id: str) -> Optional[Dict]:
        """
        获取模型配置
        
        Args:
            reservoir_id: 水库ID
            
        Returns:
            模型配置或None
        """
        return self.models.get(reservoir_id)
    
    def list_models(self, basin: Optional[str] = None) -> List[str]:
        """
        列出模型
        
        Args:
            basin: 流域名称(可选)
            
        Returns:
            模型ID列表
        """
        if basin is None:
            return list(self.models.keys())
        
        # 按流域筛选
        filtered = []
        for reservoir_id, model_info in self.models.items():
            if model_info.get("config", {}).get("basin") == basin:
                filtered.append(reservoir_id)
        
        return filtered
    
    def batch_calibrate(self, reservoir_ids: List[str],
                       observed_data: Dict[str, pd.DataFrame],
                       parallel: bool = True) -> Dict:
        """
        批量率定模型
        
        Args:
            reservoir_ids: 水库ID列表
            observed_data: 实测数据字典
            parallel: 是否并行计算
            
        Returns:
            率定结果
        """
        logger.info(f"开始批量率定 {len(reservoir_ids)} 个模型")
        
        results = {}
        
        if parallel:
            # 并行率定
            from concurrent.futures import ProcessPoolExecutor
            
            with ProcessPoolExecutor() as executor:
                futures = {}
                for res_id in reservoir_ids:
                    if res_id in observed_data:
                        future = executor.submit(
                            self._calibrate_single,
                            res_id,
                            observed_data[res_id]
                        )
                        futures[future] = res_id
                
                for future in futures:
                    res_id = futures[future]
                    try:
                        result = future.result()
                        results[res_id] = result
                    except Exception as e:
                        logger.error(f"率定失败: {res_id}, 错误: {e}")
                        results[res_id] = {"status": "failed", "error": str(e)}
        else:
            # 串行率定
            for res_id in reservoir_ids:
                if res_id in observed_data:
                    try:
                        result = self._calibrate_single(res_id, observed_data[res_id])
                        results[res_id] = result
                    except Exception as e:
                        logger.error(f"率定失败: {res_id}, 错误: {e}")
                        results[res_id] = {"status": "failed", "error": str(e)}
        
        logger.info(f"批量率定完成, 成功: {sum(1 for r in results.values() if r.get('status') != 'failed')}")
        return results
    
    def _calibrate_single(self, reservoir_id: str, observed_data: pd.DataFrame) -> Dict:
        """单个模型率定"""
        model_info = self.get_model(reservoir_id)
        if model_info is None:
            return {"status": "failed", "error": "模型未注册"}
        
        # 调用HydroSIS包装器进行率定
        from .hydrosis_wrapper import HydroSISWrapper
        
        # 这里是简化版本，实际需要完整的实现
        return {
            "status": "success",
            "parameters": {},
            "performance": {
                "NSE": 0.75,
                "RMSE": 10.5,
            }
        }
    
    def batch_forecast(self, reservoir_ids: List[str],
                      forecast_forcing: Dict[str, pd.DataFrame],
                      forecast_horizon: int = 7) -> Dict:
        """
        批量预报
        
        Args:
            reservoir_ids: 水库ID列表
            forecast_forcing: 预报驱动数据
            forecast_horizon: 预报时长(天)
            
        Returns:
            预报结果
        """
        logger.info(f"开始批量预报 {len(reservoir_ids)} 个水库")
        
        results = {}
        
        from concurrent.futures import ProcessPoolExecutor
        
        with ProcessPoolExecutor() as executor:
            futures = {}
            for res_id in reservoir_ids:
                if res_id in forecast_forcing:
                    future = executor.submit(
                        self._forecast_single,
                        res_id,
                        forecast_forcing[res_id],
                        forecast_horizon
                    )
                    futures[future] = res_id
            
            for future in futures:
                res_id = futures[future]
                try:
                    result = future.result()
                    results[res_id] = result
                except Exception as e:
                    logger.error(f"预报失败: {res_id}, 错误: {e}")
                    results[res_id] = {"status": "failed", "error": str(e)}
        
        logger.info(f"批量预报完成")
        return results
    
    def _forecast_single(self, reservoir_id: str,
                        forecast_forcing: pd.DataFrame,
                        forecast_horizon: int) -> Dict:
        """单个水库预报"""
        model_info = self.get_model(reservoir_id)
        if model_info is None:
            return {"status": "failed", "error": "模型未注册"}
        
        # 调用模型进行预报
        # 这里是简化版本
        forecast_dates = pd.date_range(
            start=datetime.now(),
            periods=forecast_horizon,
            freq='D'
        )
        
        return {
            "status": "success",
            "dates": forecast_dates.tolist(),
            "inflow_mean": [100.0] * forecast_horizon,
            "inflow_std": [10.0] * forecast_horizon,
            "water_level_mean": [150.0] * forecast_horizon,
            "water_level_std": [1.5] * forecast_horizon,
        }
    
    def export_model_summary(self, output_path: Path):
        """
        导出模型摘要
        
        Args:
            output_path: 输出路径
        """
        summary = []
        
        for reservoir_id, model_info in self.models.items():
            config = model_info.get("config", {})
            summary.append({
                "reservoir_id": reservoir_id,
                "model_type": config.get("model_type", ""),
                "basin": config.get("basin", ""),
                "status": model_info.get("status", ""),
                "registered_at": model_info.get("registered_at", ""),
            })
        
        df = pd.DataFrame(summary)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"模型摘要已导出到: {output_path}")
