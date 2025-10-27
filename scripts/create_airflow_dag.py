#!/usr/bin/env python
"""
创建Airflow DAG文件
"""
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from water_resource_system.data_acquisition.scheduler import DataScheduler


def main():
    """主函数"""
    scheduler = DataScheduler()
    
    # 设置Airflow DAGs目录
    airflow_home = Path.home() / "airflow"
    dags_dir = airflow_home / "dags"
    dags_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存DAG文件
    output_path = dags_dir / "water_resource_data_collection.py"
    scheduler.save_airflow_dag(output_path)
    
    print(f"✅ Airflow DAG已创建: {output_path}")
    print(f"📂 DAG目录: {dags_dir}")
    print("\n下一步:")
    print("1. 启动Airflow: airflow webserver -p 8080 &")
    print("2. 启动调度器: airflow scheduler &")
    print("3. 访问: http://localhost:8080")


if __name__ == "__main__":
    main()
