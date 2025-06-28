#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from qlib.utils import fname_to_code, code_to_fname
import clickhouse_connect
from typing import List, Set
from tqdm import tqdm
import shutil

class DumpDataFromClickHouse:
    """从ClickHouse数据库导出数据到qlib格式"""
    
    INSTRUMENTS_START_FIELD = "start_datetime"
    INSTRUMENTS_END_FIELD = "end_datetime"
    CALENDARS_DIR_NAME = "calendars"
    FEATURES_DIR_NAME = "features"
    INSTRUMENTS_DIR_NAME = "instruments"
    DUMP_FILE_SUFFIX = ".bin"
    DAILY_FORMAT = "%Y-%m-%d"
    INSTRUMENTS_SEP = "\t"
    INSTRUMENTS_FILE_NAME = "all.txt"
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8123,
        username: str = "nero",
        password: str = "ck7981",
        database: str = "tushare",
        qlib_dir: str = "/home/nero/.qlib/qlib_data/tushare_data",
        start_date: str = "2010-02-01",
        end_date: str = "2025-06-20",
        max_workers: int = 32,
    ):
        try:
            self.client = clickhouse_connect.get_client(
                    host=host, port=port, username=username, password=password, database=database
                )
            logger.info(f"成功连接到ClickHouse: {host}:{port}")
        except Exception as e:
            logger.error(f"连接ClickHouse失败: {e}")
            raise
        self.qlib_dir = Path(qlib_dir).expanduser()
        self.start_date = start_date
        self.end_date = end_date
        self.max_workers = max_workers
        
        self._calendars_dir = self.qlib_dir.joinpath(self.CALENDARS_DIR_NAME)
        self._features_dir = self.qlib_dir.joinpath(self.FEATURES_DIR_NAME)
        self._instruments_dir = self.qlib_dir.joinpath(self.INSTRUMENTS_DIR_NAME)
        
        self.qlib_dir.mkdir(parents=True, exist_ok=True)
        self._calendars_dir.mkdir(parents=True, exist_ok=True)
        self._features_dir.mkdir(parents=True, exist_ok=True)
        self._instruments_dir.mkdir(parents=True, exist_ok=True)
        
        # 在 __init__ 里保存连接参数
        self._client_params = dict(
            host=host, port=port, username=username, password=password, database=database
        )
    
    def _format_datetime(self, datetime_d) -> str:
        """格式化日期时间"""
        datetime_d = pd.Timestamp(datetime_d)
        return datetime_d.strftime(self.DAILY_FORMAT)
    
    def get_stock_list(self) -> List[str]:
        """获取股票代码列表"""
        query = f"""
        SELECT DISTINCT ts_code 
        FROM daily 
        WHERE trade_date >= '{self.start_date}' AND trade_date <= '{self.end_date}'
        ORDER BY ts_code
        """
        result = self.client.query(query)
        return [row[0] for row in result.result_rows]
    
    def get_index_list(self) -> List[str]:
        """获取指数代码列表"""
        query = f"""
        SELECT DISTINCT ts_code 
        FROM index_daily 
        WHERE trade_date >= '{self.start_date}' AND trade_date <= '{self.end_date}'
        ORDER BY ts_code
        """
        result = self.client.query(query)
        return [row[0] for row in result.result_rows]
    
    def get_calendar_dates(self) -> Set[pd.Timestamp]:
        """获取所有交易日期"""
        query = f"""
        SELECT DISTINCT trade_date 
        FROM daily 
        WHERE trade_date >= '{self.start_date}' AND trade_date <= '{self.end_date}'
        ORDER BY trade_date
        """
        result = self.client.query(query)
        return set(pd.Timestamp(row[0]) for row in result.result_rows)
    
    def get_stock_data(self, ts_code: str, client=None) -> pd.DataFrame:
        """获取单只股票的所有数据"""
        if client is None:
            client = self.client
        
        query = f"""
        SELECT 
            sd.ts_code,
            sd.trade_date,
            sd.open * COALESCE(af.adj_factor, 1) as open,
            sd.high * COALESCE(af.adj_factor, 1) as high,
            sd.low * COALESCE(af.adj_factor, 1) as low,
            sd.close * COALESCE(af.adj_factor, 1) as close,
            sd.change * COALESCE(af.adj_factor, 1) as change,
            sd.pct_chg,
            sd.vol as volume,
            sd.amount,
            COALESCE(af.adj_factor, 1) as factor,
            db.turnover_rate,
            db.turnover_rate_f,
            db.volume_ratio,
            db.pe,
            db.pe_ttm,
            db.pb,
            db.ps,
            db.ps_ttm,
            db.dv_ratio,
            db.dv_ttm,
            db.total_share,
            db.float_share,
            db.free_share,
            db.total_mv,
            db.circ_mv,
            db.limit_status,
            mf.buy_sm_vol,
            mf.buy_sm_amount,
            mf.sell_sm_vol,
            mf.sell_sm_amount,
            mf.buy_md_vol,
            mf.buy_md_amount,
            mf.sell_md_vol,
            mf.sell_md_amount,
            mf.buy_lg_vol,
            mf.buy_lg_amount,
            mf.sell_lg_vol,
            mf.sell_lg_amount,
            mf.buy_elg_vol,
            mf.buy_elg_amount,
            mf.sell_elg_vol,
            mf.sell_elg_amount,
            mf.net_mf_vol,
            mf.net_mf_amount,
            mf.trade_count
        FROM daily sd
        LEFT JOIN adj_factor af ON sd.ts_code = af.ts_code AND sd.trade_date = af.trade_date
        LEFT JOIN daily_basic db ON sd.ts_code = db.ts_code AND sd.trade_date = db.trade_date
        LEFT JOIN moneyflow mf ON sd.ts_code = mf.ts_code AND sd.trade_date = mf.trade_date
        WHERE sd.ts_code = '{ts_code}' 
        AND sd.trade_date >= '{self.start_date}' 
        AND sd.trade_date <= '{self.end_date}'
        ORDER BY sd.trade_date
        """
        
        result = client.query(query)
        columns = [
            'ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'change', 'pct_chg', 'volume', 'amount',
            'factor', 'turnover_rate', 'turnover_rate_f', 'volume_ratio', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm',
            'dv_ratio', 'dv_ttm', 'total_share', 'float_share', 'free_share', 'total_mv', 'circ_mv',
            'limit_status', 'buy_sm_vol', 'buy_sm_amount', 'sell_sm_vol', 'sell_sm_amount', 'buy_md_vol',
            'buy_md_amount', 'sell_md_vol', 'sell_md_amount', 'buy_lg_vol', 'buy_lg_amount', 'sell_lg_vol',
            'sell_lg_amount', 'buy_elg_vol', 'buy_elg_amount', 'sell_elg_vol', 'sell_elg_amount',
            'net_mf_vol', 'net_mf_amount', 'trade_count'
        ]
        
        df = pd.DataFrame(result.result_rows, columns=columns)
        if not df.empty:
            # 先转换日期类型
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            # 去重：保留最后一条记录（通常最新的数据更准确）
            df = df.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last')
            df.set_index('trade_date', inplace=True)
            df = df.drop(columns=['ts_code'])
        return df
    
    def get_index_data(self, ts_code: str, client=None) -> pd.DataFrame:
        """获取单只指数的所有数据"""
        if client is None:
            client = self.client
        
        query = f"""
        SELECT 
            ts_code,
            trade_date,
            open,
            high,
            low,
            close,
            pre_close,
            change,
            pct_chg,
            vol as volume,
            amount
        FROM index_daily
        WHERE ts_code = '{ts_code}' 
        AND trade_date >= '{self.start_date}' 
        AND trade_date <= '{self.end_date}'
        ORDER BY trade_date
        """
        
        result = client.query(query)
        columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'volume', 'amount']
        
        df = pd.DataFrame(result.result_rows, columns=columns)
        if not df.empty:
            # 先转换日期类型
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            # 去重：保留最后一条记录（通常最新的数据更准确）
            df = df.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last')
            df.set_index('trade_date', inplace=True)
            df = df.drop(columns=['ts_code'])
        return df
    
    def save_calendars(self, calendars_data: List[pd.Timestamp]):
        """保存交易日历"""
        calendars_path = self._calendars_dir.joinpath("day.txt")
        result_calendars_list = [self._format_datetime(x) for x in sorted(calendars_data)]
        np.savetxt(str(calendars_path), result_calendars_list, fmt="%s", encoding="utf-8")
    
    def save_instruments(self, instruments_data: List[str]):
        """保存股票列表"""
        instruments_path = self._instruments_dir.joinpath(self.INSTRUMENTS_FILE_NAME)
        with open(instruments_path, 'w', encoding='utf-8') as f:
            for line in instruments_data:
                f.write(line + '\n')
    
    @staticmethod
    def tscode_to_feature_dirname(ts_code: str) -> str:
        """将ts_code转换为 sz000001 格式"""
        ts_code_lower = ts_code.lower()
        if '.' in ts_code_lower:
            code_part, market_part = ts_code_lower.split('.')
            return f"{market_part}{code_part}"
        else:
            return ts_code_lower
    
    def data_merge_calendar(self, df: pd.DataFrame, calendars_list: List[pd.Timestamp]) -> pd.DataFrame:
        """将数据与交易日历对齐"""
        calendars_df = pd.DataFrame(data=calendars_list, columns=['date'])
        calendars_df['date'] = pd.to_datetime(calendars_df['date'])
        cal_df = calendars_df[
            (calendars_df['date'] >= df.index.min()) & 
            (calendars_df['date'] <= df.index.max())
        ]
        cal_df.set_index('date', inplace=True)
        r_df = df.reindex(cal_df.index)
        return r_df
    
    @staticmethod
    def get_datetime_index(df: pd.DataFrame, calendar_list: List[pd.Timestamp]) -> int:
        """获取日期在日历中的索引"""
        return calendar_list.index(df.index.min())
    
    def _data_to_bin(self, df: pd.DataFrame, calendar_list: List[pd.Timestamp], features_dir: Path, symbol: str):
        """将数据转换为二进制格式"""
        if df.empty:
            logger.warning(f"{symbol} data is empty")
            return
        
        _df = self.data_merge_calendar(df, calendar_list)
        if _df.empty:
            logger.warning(f"{symbol} data is not in calendars")
            return
        
        date_index = self.get_datetime_index(_df, calendar_list)
        for field in _df.columns:
            bin_path = features_dir.joinpath(f"{field.lower()}.day{self.DUMP_FILE_SUFFIX}")
            if field not in _df.columns:
                continue
            np.hstack([date_index, _df[field]]).astype("<f").tofile(str(bin_path.resolve()))
    
    def _dump_single_stock(self, ts_code: str, calendar_list: List[pd.Timestamp]) -> tuple:
        """处理单只股票数据"""
        try:
            client = clickhouse_connect.get_client(**self._client_params)
            df = self.get_stock_data(ts_code, client)
            if df.empty:
                logger.warning(f"{ts_code} data is empty")
                return None, None, None
            # 使用统一的 sz000001 格式
            features_dir_name = self.tscode_to_feature_dirname(ts_code)
            features_dir = self._features_dir.joinpath(features_dir_name)
            features_dir.mkdir(parents=True, exist_ok=True)
            
            self._data_to_bin(df, calendar_list, features_dir, ts_code)
            
            start_date = self._format_datetime(df.index.min())
            end_date = self._format_datetime(df.index.max())
            
            # instruments文件中的code也用 sz000001 格式
            return features_dir_name.upper(), start_date, end_date
        except Exception as e:
            logger.error(f"Error processing {ts_code}: {e}")
            return None, None, None
    
    def _dump_single_index(self, ts_code: str, calendar_list: List[pd.Timestamp]) -> tuple:
        """处理单只指数数据"""
        try:
            client = clickhouse_connect.get_client(**self._client_params)
            df = self.get_index_data(ts_code, client)
            if df.empty:
                logger.warning(f"{ts_code} data is empty")
                return None, None, None
            features_dir_name = self.tscode_to_feature_dirname(ts_code)
            features_dir = self._features_dir.joinpath(features_dir_name)
            features_dir.mkdir(parents=True, exist_ok=True)
            
            self._data_to_bin(df, calendar_list, features_dir, ts_code)
            
            start_date = self._format_datetime(df.index.min())
            end_date = self._format_datetime(df.index.max())
            
            return features_dir_name.upper(), start_date, end_date
        except Exception as e:
            logger.error(f"Error processing {ts_code}: {e}")
            return None, None, None
    
    def _clear_existing_data(self):
        """清理已存在的数据文件"""
        logger.info("清理已存在的数据文件...")
        try:
            # 删除 features 目录下的所有文件夹
            if self._features_dir.exists():
                for item in self._features_dir.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
            
            # 删除 calendars 目录下的文件
            if self._calendars_dir.exists():
                for item in self._calendars_dir.iterdir():
                    if item.is_file():
                        item.unlink()
            
            # 删除 instruments 目录下的文件
            if self._instruments_dir.exists():
                for item in self._instruments_dir.iterdir():
                    if item.is_file():
                        item.unlink()
            
            logger.info("清理目录完成")
        except Exception as e:
            logger.error(f"清理数据时出错: {e}")
            raise
    
    def dump(self):
        """主导出函数"""
        logger.info("开始从ClickHouse导出数据...")
        
        # 清理已存在的数据
        self._clear_existing_data()
        
        # 获取股票列表
        logger.info("获取股票列表...")
        stock_list = self.get_stock_list()
        logger.info(f"共找到 {len(stock_list)} 只股票")
        
        # 获取指数列表
        logger.info("获取指数列表...")
        index_list = self.get_index_list()
        logger.info(f"共找到 {len(index_list)} 只指数")
        
        # 获取交易日历
        logger.info("获取交易日历...")
        calendar_dates = self.get_calendar_dates()
        calendar_list = sorted(calendar_dates)
        self.save_calendars(calendar_list)
        logger.info(f"共 {len(calendar_list)} 个交易日")
        
        # 导出股票数据
        logger.info("开始导出股票数据...")
        instruments_data = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._dump_single_stock, ts_code, calendar_list): ts_code 
                      for ts_code in stock_list}
            
            with tqdm(total=len(futures), desc="股票导出进度") as pbar:
                for future in as_completed(futures):
                    code, start_date, end_date = future.result()
                    if code:
                        instruments_data.append(f"{code}{self.INSTRUMENTS_SEP}{start_date}{self.INSTRUMENTS_SEP}{end_date}")
                    pbar.update(1)
        
        # 导出指数数据
        logger.info("开始导出指数数据...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._dump_single_index, ts_code, calendar_list): ts_code 
                      for ts_code in index_list}
            
            with tqdm(total=len(futures), desc="指数导出进度") as pbar:
                for future in as_completed(futures):
                    code, start_date, end_date = future.result()
                    if code:
                        instruments_data.append(f"{code}{self.INSTRUMENTS_SEP}{start_date}{self.INSTRUMENTS_SEP}{end_date}")
                    pbar.update(1)
        
        # 保存股票列表
        self.save_instruments(instruments_data)
        logger.info(f"数据导出完成，共处理 {len(instruments_data)} 只股票和指数")

def main():
    """测试函数"""
    dumper = DumpDataFromClickHouse(password="ck7981")
    
    # 先测试查询少量数据
    logger.info("测试查询股票数量...")
    try:
        stock_list = dumper.get_stock_list()
        logger.info(f"共找到 {len(stock_list)} 只股票")
        if len(stock_list) > 0:
            logger.info(f"前5只股票: {stock_list[:5]}")
        
        index_list = dumper.get_index_list()
        logger.info(f"共找到 {len(index_list)} 只指数")
        if len(index_list) > 0:
            logger.info(f"前5只指数: {index_list[:5]}")
        
        if len(stock_list) == 0 and len(index_list) == 0:
            logger.warning("未找到股票和指数数据，请检查数据库表是否存在数据")
            return
            
        # 完整导出
        dumper.dump()
        
        # 验证生成的文件
        qlib_dir = Path("/home/nero/.qlib/qlib_data/tushare_data")
        logger.info(f"验证生成的文件:")
        logger.info(f"日历文件: {qlib_dir / 'calendars' / 'day.txt'}")
        logger.info(f"股票列表: {qlib_dir / 'instruments' / 'all.txt'}")
        
        features_dir = qlib_dir / "features"
        if features_dir.exists():
            stock_dirs = list(features_dir.iterdir())
            logger.info(f"特征数据目录数量: {len(stock_dirs)}")
            if stock_dirs:
                sample_dir = stock_dirs[0]
                bin_files = list(sample_dir.glob("*.bin"))
                logger.info(f"示例股票 {sample_dir.name} 的特征文件数量: {len(bin_files)}")
                logger.info(f"特征文件: {[f.name for f in bin_files[:5]]}")
    except Exception as e:
        logger.error(f"执行过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
