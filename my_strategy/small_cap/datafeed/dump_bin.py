import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import fire

# 添加 qlib 路径
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from scripts.dump_bin import DumpDataBase


class JuKuanDataDumper(DumpDataBase):
    """聚宽 H5 数据转换为 qlib bin 格式"""

    def __init__(
        self,
        h5_file_path: str,
        qlib_dir: str,
        freq: str = "day",
        max_workers: int = 16,
        instruments_file_name: str = "399101.txt",  # 新增参数
    ):
        # 基础参数
        self.h5_file_path = Path(h5_file_path).expanduser().resolve()
        self.qlib_dir = Path(qlib_dir).expanduser().resolve()
        self.freq = freq.lower()
        self.works = max_workers

        # 补齐父类里原本由 __init__ 设置的关键属性
        self.calendar_format = self.DAILY_FORMAT if self.freq == "day" else self.HIGH_FREQ_FORMAT
        self.date_field_name = "date"
        self.symbol_field_name = "symbol"
        # 关键修复：排除非数值字段
        self._exclude_fields = ("symbol", "date")  # 排除股票代码和日期
        self._include_fields = tuple()
        self._mode = self.ALL_MODE   # 不是更新模式

        # 覆盖父类的默认文件名
        self.INSTRUMENTS_FILE_NAME = instruments_file_name

        # 目录
        self._calendars_dir = self.qlib_dir.joinpath(self.CALENDARS_DIR_NAME)
        self._features_dir = self.qlib_dir.joinpath(self.FEATURES_DIR_NAME)
        self._instruments_dir = self.qlib_dir.joinpath(self.INSTRUMENTS_DIR_NAME)

        # 确保根目录存在
        self.qlib_dir.mkdir(parents=True, exist_ok=True)

        # 读取数据
        self.df = self._load_h5_data()

    def _load_h5_data(self) -> pd.DataFrame:
        print("读取 H5 数据...")
        df = pd.read_hdf(self.h5_file_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['code', 'date'])
        df['factor'] = 1.0
        column_mapping = {
            'code': 'symbol',
            'date': 'date',
            'open': 'open',
            'close': 'close',
            'high': 'high',
            'low': 'low',
            'volume': 'volume',
            'factor': 'factor',
            'market_cap': 'market_cap',                           # 总市值
            'circulating_market_cap': 'circulating_market_cap'   # 流通市值
        }
        needed_cols = list(column_mapping.keys())
        df = df[needed_cols].rename(columns=column_mapping)
        print(f"数据加载完成，共 {len(df)} 行数据")
        return df

    def _get_symbol_list(self):
        return sorted(self.df[self.symbol_field_name].unique())

    def _get_date_list(self):
        return sorted(self.df[self.date_field_name].unique())

    def _dump_calendars(self):
        print("生成交易日历...")
        calendars_list = self._get_date_list()
        self.save_calendars(calendars_list)
        return calendars_list

    def _dump_instruments(self):
        """生成股票列表文件（优化版：避免重复过滤）"""
        print("生成股票列表...")
        instruments_data = []
        
        # 使用 groupby 一次性获取每个股票的日期范围，避免重复过滤
        symbol_date_ranges = (
            self.df.groupby(self.symbol_field_name)[self.date_field_name]
            .agg(['min', 'max'])
            .reset_index()
        )
        
        for _, row in symbol_date_ranges.iterrows():
            symbol = row[self.symbol_field_name]
            start_str = self._format_datetime(row['min'])
            end_str = self._format_datetime(row['max'])
            instruments_data.append(f"{symbol.upper()}\t{start_str}\t{end_str}")
        
        self.save_instruments(instruments_data)

    def _dump_features(self):
        """生成特征数据（优化版：减少重复分组）"""
        print("生成特征数据...")
        calendars_list = self._get_date_list()
        from qlib.utils import code_to_fname
        
        # 按股票分组，避免重复过滤大DataFrame
        grouped = self.df.groupby(self.symbol_field_name, group_keys=False)
        for symbol, symbol_df in tqdm(grouped, desc="处理股票数据"):
            symbol_dir = self._features_dir.joinpath(code_to_fname(symbol.lower()))
            symbol_dir.mkdir(parents=True, exist_ok=True)
            # 不要预先设 index，让 _data_to_bin 内部处理
            self._data_to_bin(symbol_df.copy(), calendars_list, symbol_dir)

    def dump(self):
        print("开始转换聚宽数据到 qlib 格式...")
        self.qlib_dir.mkdir(parents=True, exist_ok=True)
        self._dump_calendars()
        self._dump_instruments()
        self._dump_features()
        print(f"转换完成！数据保存在: {self.qlib_dir}")


def convert_jukuan_data(
    h5_file_path: str = "/mnt/c/nero/quant/data/jukuan/small_cap_olhcv_data.h5",
    qlib_dir: str = "/home/nero/.qlib/qlib_data/jukuan_data",
    freq: str = "day",
    max_workers: int = 16,
    instruments_file_name: str = "399101.txt",  # 新增参数
):
    """
    转换聚宽 H5 数据为 qlib bin 格式
    
    Parameters
    ----------
    h5_file_path: str
        聚宽 H5 文件路径
    qlib_dir: str
        qlib 数据输出目录
    freq: str
        频率
    max_workers: int
        线程数
    instruments_file_name: str
        股票列表文件名，默认为 399101.txt (中小板指)
    """
    dumper = JuKuanDataDumper(
        h5_file_path=h5_file_path,
        qlib_dir=qlib_dir,
        freq=freq,
        max_workers=max_workers,
        instruments_file_name=instruments_file_name,
    )
    dumper.dump()


if __name__ == "__main__":
    fire.Fire(convert_jukuan_data)
