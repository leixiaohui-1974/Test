#!/usr/bin/env python
"""
初始化数据库
创建必要的表和索引
"""
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from water_resource_system.config import DATABASE_CONFIGS
from water_resource_system.database import PostgreSQLManager, InfluxDBManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_postgresql():
    """初始化PostgreSQL数据库"""
    logger.info("=" * 60)
    logger.info("初始化PostgreSQL数据库")
    logger.info("=" * 60)
    
    try:
        # 创建数据库管理器
        db = PostgreSQLManager(DATABASE_CONFIGS["postgresql"])
        db.connect()
        
        # 启用PostGIS扩展
        logger.info("启用PostGIS扩展...")
        db.cursor.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
        db.conn.commit()
        
        # 创建水库信息表
        logger.info("创建水库信息表...")
        db.create_reservoir_table()
        
        # 创建流域信息表
        logger.info("创建流域信息表...")
        db.cursor.execute("""
        CREATE TABLE IF NOT EXISTS watersheds (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            code VARCHAR(50) UNIQUE,
            boundary GEOMETRY(MULTIPOLYGON, 4326),
            area_km2 FLOAT,
            parent_id INTEGER REFERENCES watersheds(id),
            level INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_watersheds_boundary 
        ON watersheds USING GIST(boundary);
        """)
        db.conn.commit()
        
        # 创建气象站表
        logger.info("创建气象站表...")
        db.cursor.execute("""
        CREATE TABLE IF NOT EXISTS meteorology_stations (
            id SERIAL PRIMARY KEY,
            station_id VARCHAR(50) UNIQUE NOT NULL,
            name VARCHAR(100),
            location GEOMETRY(POINT, 4326),
            elevation FLOAT,
            province VARCHAR(50),
            station_type VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_met_stations_location 
        ON meteorology_stations USING GIST(location);
        """)
        db.conn.commit()
        
        # 创建水文站表
        logger.info("创建水文站表...")
        db.cursor.execute("""
        CREATE TABLE IF NOT EXISTS hydrology_stations (
            id SERIAL PRIMARY KEY,
            station_id VARCHAR(50) UNIQUE NOT NULL,
            name VARCHAR(100),
            location GEOMETRY(POINT, 4326),
            river_name VARCHAR(100),
            drainage_area FLOAT,
            station_type VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_hydro_stations_location 
        ON hydrology_stations USING GIST(location);
        """)
        db.conn.commit()
        
        # 插入中国主要流域信息
        logger.info("插入流域数据...")
        from water_resource_system.config import CHINA_BASINS
        
        for basin_name, basin_info in CHINA_BASINS.items():
            db.cursor.execute("""
            INSERT INTO watersheds (name, code, area_km2, level)
            VALUES (%s, %s, %s, 1)
            ON CONFLICT (code) DO NOTHING;
            """, (basin_name, basin_info["code"], basin_info["area"]))
        
        db.conn.commit()
        
        logger.info("✅ PostgreSQL数据库初始化完成")
        
        db.close()
        
    except Exception as e:
        logger.error(f"❌ PostgreSQL初始化失败: {e}")


def init_influxdb():
    """初始化InfluxDB数据库"""
    logger.info("=" * 60)
    logger.info("初始化InfluxDB数据库")
    logger.info("=" * 60)
    
    try:
        # 创建数据库管理器
        db = InfluxDBManager(DATABASE_CONFIGS["influxdb"])
        db.connect()
        
        # 创建保留策略
        logger.info("创建数据保留策略...")
        
        # 原始数据保留3年
        db.client.create_retention_policy(
            name='raw_data',
            duration='156w',  # 3年
            replication='1',
            database=db.config["database"],
            default=True
        )
        
        # 创建连续查询用于数据降采样
        logger.info("创建连续查询...")
        
        # 小时平均值，保留10年
        db.client.query("""
        CREATE CONTINUOUS QUERY "cq_hourly_mean" ON "water_monitoring"
        BEGIN
          SELECT mean(*) INTO "water_monitoring"."ten_years".:MEASUREMENT
          FROM "water_monitoring"."raw_data"./.*/ 
          GROUP BY time(1h), *
        END
        """, method='POST')
        
        # 日平均值，永久保留
        db.client.query("""
        CREATE CONTINUOUS QUERY "cq_daily_mean" ON "water_monitoring"
        BEGIN
          SELECT mean(*) INTO "water_monitoring"."forever".:MEASUREMENT
          FROM "water_monitoring"."raw_data"./.*/ 
          GROUP BY time(1d), *
        END
        """, method='POST')
        
        logger.info("✅ InfluxDB数据库初始化完成")
        
        db.close()
        
    except Exception as e:
        logger.error(f"❌ InfluxDB初始化失败: {e}")
        logger.info("注意: 某些错误(如已存在的策略)可以忽略")


def verify_connections():
    """验证数据库连接"""
    logger.info("=" * 60)
    logger.info("验证数据库连接")
    logger.info("=" * 60)
    
    # 验证PostgreSQL
    try:
        db = PostgreSQLManager(DATABASE_CONFIGS["postgresql"])
        db.connect()
        db.cursor.execute("SELECT version();")
        version = db.cursor.fetchone()
        logger.info(f"✅ PostgreSQL连接正常: {version['version']}")
        db.close()
    except Exception as e:
        logger.error(f"❌ PostgreSQL连接失败: {e}")
    
    # 验证InfluxDB
    try:
        db = InfluxDBManager(DATABASE_CONFIGS["influxdb"])
        db.connect()
        version = db.client.ping()
        logger.info(f"✅ InfluxDB连接正常")
        db.close()
    except Exception as e:
        logger.error(f"❌ InfluxDB连接失败: {e}")


def print_summary():
    """打印总结"""
    logger.info("\n")
    logger.info("=" * 60)
    logger.info("数据库初始化总结")
    logger.info("=" * 60)
    logger.info("\nPostgreSQL表:")
    logger.info("  ✅ reservoirs - 水库基础信息")
    logger.info("  ✅ watersheds - 流域信息")
    logger.info("  ✅ meteorology_stations - 气象站")
    logger.info("  ✅ hydrology_stations - 水文站")
    logger.info("\nInfluxDB measurement:")
    logger.info("  ✅ reservoir_water_level - 水位监测")
    logger.info("  ✅ meteorology - 气象观测")
    logger.info("\n数据保留策略:")
    logger.info("  📋 原始数据: 3年")
    logger.info("  📋 小时数据: 10年")
    logger.info("  📋 日数据: 永久")
    logger.info("\n下一步:")
    logger.info("  1. 导入水库基础数据")
    logger.info("  2. 导入站点数据")
    logger.info("  3. 开始数据采集")
    logger.info("\n")


def main():
    """主函数"""
    logger.info("\n")
    logger.info("=" * 60)
    logger.info("水资源监测系统 - 数据库初始化")
    logger.info("=" * 60)
    logger.info("\n")
    
    # 验证连接
    verify_connections()
    
    # 初始化PostgreSQL
    init_postgresql()
    
    # 初始化InfluxDB
    init_influxdb()
    
    # 打印总结
    print_summary()


if __name__ == "__main__":
    main()
