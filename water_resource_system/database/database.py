"""
数据库操作模块
支持PostgreSQL(空间数据)和InfluxDB(时序数据)
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class PostgreSQLManager:
    """PostgreSQL + PostGIS数据库管理器"""
    
    def __init__(self, config: Dict):
        """
        初始化数据库连接
        
        Args:
            config: 数据库配置
        """
        self.config = config
        self.conn = None
        self.cursor = None
    
    def connect(self):
        """建立数据库连接"""
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            
            self.conn = psycopg2.connect(
                host=self.config["host"],
                port=self.config["port"],
                database=self.config["database"],
                user=self.config["username"],
                password=self.config["password"]
            )
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            
            logger.info("PostgreSQL连接成功")
            
        except ImportError:
            logger.error("psycopg2库未安装，请运行: pip install psycopg2-binary")
        except Exception as e:
            logger.error(f"PostgreSQL连接失败: {e}")
    
    def close(self):
        """关闭数据库连接"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("PostgreSQL连接已关闭")
    
    def create_reservoir_table(self):
        """创建水库信息表"""
        sql = """
        CREATE TABLE IF NOT EXISTS reservoirs (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            code VARCHAR(50) UNIQUE,
            location GEOMETRY(POINT, 4326),
            watershed_id INTEGER,
            total_capacity FLOAT,
            normal_level FLOAT,
            dead_level FLOAT,
            province VARCHAR(50),
            basin VARCHAR(50),
            construction_year INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_reservoirs_location 
        ON reservoirs USING GIST(location);
        
        CREATE INDEX IF NOT EXISTS idx_reservoirs_basin 
        ON reservoirs(basin);
        """
        
        try:
            self.cursor.execute(sql)
            self.conn.commit()
            logger.info("水库信息表创建成功")
        except Exception as e:
            logger.error(f"创建表失败: {e}")
            self.conn.rollback()
    
    def insert_reservoir(self, reservoir_data: Dict) -> Optional[int]:
        """
        插入水库信息
        
        Args:
            reservoir_data: 水库数据
            
        Returns:
            插入的记录ID
        """
        sql = """
        INSERT INTO reservoirs (name, code, location, total_capacity, 
                               normal_level, dead_level, province, basin)
        VALUES (%(name)s, %(code)s, ST_SetSRID(ST_MakePoint(%(lon)s, %(lat)s), 4326),
                %(total_capacity)s, %(normal_level)s, %(dead_level)s, 
                %(province)s, %(basin)s)
        RETURNING id;
        """
        
        try:
            self.cursor.execute(sql, reservoir_data)
            reservoir_id = self.cursor.fetchone()['id']
            self.conn.commit()
            logger.info(f"插入水库: {reservoir_data['name']}, ID={reservoir_id}")
            return reservoir_id
        except Exception as e:
            logger.error(f"插入水库失败: {e}")
            self.conn.rollback()
            return None
    
    def query_reservoirs_by_basin(self, basin: str) -> List[Dict]:
        """
        按流域查询水库
        
        Args:
            basin: 流域名称
            
        Returns:
            水库列表
        """
        sql = """
        SELECT id, name, code, 
               ST_X(location) as longitude, 
               ST_Y(location) as latitude,
               total_capacity, normal_level, dead_level, province
        FROM reservoirs
        WHERE basin = %s
        ORDER BY total_capacity DESC;
        """
        
        try:
            self.cursor.execute(sql, (basin,))
            results = self.cursor.fetchall()
            logger.info(f"查询到 {len(results)} 个水库")
            return results
        except Exception as e:
            logger.error(f"查询失败: {e}")
            return []
    
    def query_reservoirs_in_bbox(self, west: float, south: float,
                                 east: float, north: float) -> List[Dict]:
        """
        按边界框查询水库
        
        Args:
            west, south, east, north: 边界框坐标
            
        Returns:
            水库列表
        """
        sql = """
        SELECT id, name, code,
               ST_X(location) as longitude,
               ST_Y(location) as latitude,
               total_capacity, province, basin
        FROM reservoirs
        WHERE ST_Within(
            location,
            ST_MakeEnvelope(%s, %s, %s, %s, 4326)
        );
        """
        
        try:
            self.cursor.execute(sql, (west, south, east, north))
            results = self.cursor.fetchall()
            logger.info(f"区域内查询到 {len(results)} 个水库")
            return results
        except Exception as e:
            logger.error(f"查询失败: {e}")
            return []


class InfluxDBManager:
    """InfluxDB时序数据管理器"""
    
    def __init__(self, config: Dict):
        """
        初始化InfluxDB连接
        
        Args:
            config: 数据库配置
        """
        self.config = config
        self.client = None
    
    def connect(self):
        """建立数据库连接"""
        try:
            from influxdb import InfluxDBClient
            
            self.client = InfluxDBClient(
                host=self.config["host"],
                port=self.config["port"],
                username=self.config["username"],
                password=self.config["password"],
                database=self.config["database"]
            )
            
            # 创建数据库(如果不存在)
            self.client.create_database(self.config["database"])
            
            logger.info("InfluxDB连接成功")
            
        except ImportError:
            logger.error("influxdb库未安装，请运行: pip install influxdb")
        except Exception as e:
            logger.error(f"InfluxDB连接失败: {e}")
    
    def close(self):
        """关闭数据库连接"""
        if self.client:
            self.client.close()
        logger.info("InfluxDB连接已关闭")
    
    def write_water_level(self, reservoir_id: str, 
                         timestamp: datetime,
                         water_level: float,
                         storage: Optional[float] = None,
                         inflow: Optional[float] = None,
                         outflow: Optional[float] = None,
                         data_source: str = "observation"):
        """
        写入水位数据
        
        Args:
            reservoir_id: 水库ID
            timestamp: 时间戳
            water_level: 水位(m)
            storage: 蓄水量(万m³)
            inflow: 入库流量(m³/s)
            outflow: 出库流量(m³/s)
            data_source: 数据源
        """
        json_body = [
            {
                "measurement": "reservoir_water_level",
                "tags": {
                    "reservoir_id": reservoir_id,
                    "data_source": data_source
                },
                "time": timestamp.isoformat(),
                "fields": {
                    "water_level": water_level,
                }
            }
        ]
        
        if storage is not None:
            json_body[0]["fields"]["storage"] = storage
        if inflow is not None:
            json_body[0]["fields"]["inflow"] = inflow
        if outflow is not None:
            json_body[0]["fields"]["outflow"] = outflow
        
        try:
            self.client.write_points(json_body)
        except Exception as e:
            logger.error(f"写入水位数据失败: {e}")
    
    def batch_write_water_level(self, data_points: List[Dict]):
        """
        批量写入水位数据
        
        Args:
            data_points: 数据点列表
        """
        json_body = []
        
        for point in data_points:
            fields = {"water_level": point["water_level"]}
            
            # 添加可选字段
            for key in ["storage", "inflow", "outflow"]:
                if key in point:
                    fields[key] = point[key]
            
            json_body.append({
                "measurement": "reservoir_water_level",
                "tags": {
                    "reservoir_id": point["reservoir_id"],
                    "data_source": point.get("data_source", "observation")
                },
                "time": point["timestamp"].isoformat(),
                "fields": fields
            })
        
        try:
            self.client.write_points(json_body)
            logger.info(f"批量写入 {len(json_body)} 条水位数据")
        except Exception as e:
            logger.error(f"批量写入失败: {e}")
    
    def query_water_level(self, reservoir_id: str,
                         start_time: datetime,
                         end_time: datetime) -> List[Dict]:
        """
        查询水位数据
        
        Args:
            reservoir_id: 水库ID
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            数据列表
        """
        query = f"""
        SELECT * FROM reservoir_water_level
        WHERE reservoir_id = '{reservoir_id}'
        AND time >= '{start_time.isoformat()}'
        AND time <= '{end_time.isoformat()}'
        ORDER BY time ASC;
        """
        
        try:
            result = self.client.query(query)
            points = list(result.get_points())
            logger.info(f"查询到 {len(points)} 条水位数据")
            return points
        except Exception as e:
            logger.error(f"查询失败: {e}")
            return []
    
    def write_meteorology(self, station_id: str,
                         timestamp: datetime,
                         variable: str,
                         value: float):
        """
        写入气象数据
        
        Args:
            station_id: 站点ID
            timestamp: 时间戳
            variable: 变量名
            value: 数值
        """
        json_body = [
            {
                "measurement": "meteorology",
                "tags": {
                    "station_id": station_id,
                    "variable": variable
                },
                "time": timestamp.isoformat(),
                "fields": {
                    "value": value
                }
            }
        ]
        
        try:
            self.client.write_points(json_body)
        except Exception as e:
            logger.error(f"写入气象数据失败: {e}")
