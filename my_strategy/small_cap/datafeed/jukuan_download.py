# %% 库导入
from jqdata import *
from jqfactor import *
import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
import warnings
warnings.filterwarnings("ignore")

# %% 配置参数
SAVE_BATCH_SIZE = 200  # 写入磁盘间隔
START_DATE = '2010-01-01'
END_DATE = '2025-09-06'
STOCK_POOL = 'ZXBZ'  # 中小板指
DATA_FILE = 'small_cap_olhcv_data.h5'
PROGRESS_FILE = 'olhcv_progress.csv'  # 进度文件
NAN_THRESHOLD = 0.5  # NaN值过滤阈值，超过50%的列将被过滤

# %% 工具函数
def delect_stop(stocks, beginDate, n=30 * 3):
    stockList = []
    beginDate = datetime.datetime.strptime(beginDate, "%Y-%m-%d")
    for stock in stocks:
        start_date = get_security_info(stock).start_date
        if start_date < (beginDate - datetime.timedelta(days=n)).date():
            stockList.append(stock)
    return stockList

def get_stock(stockPool, begin_date):
    if stockPool == 'HS300':
        stockList = get_index_stocks('000300.XSHG', begin_date)
    elif stockPool == 'ZZ500':
        stockList = get_index_stocks('399905.XSHE', begin_date)
    elif stockPool == 'ZZ800':
        stockList = get_index_stocks('399906.XSHE', begin_date)
    elif stockPool == 'CYBZ':
        stockList = get_index_stocks('399006.XSHE', begin_date)
    elif stockPool == 'ZXBZ':
        stockList = get_index_stocks('399101.XSHE', begin_date)
    elif stockPool == 'A':
        stockList = get_index_stocks('000002.XSHG', begin_date) + get_index_stocks('399107.XSHE', begin_date)
        stockList = [stock for stock in stockList if not stock.startswith(('68', '4', '8'))]
    elif stockPool == 'AA':
        stockList = get_index_stocks('000985.XSHG', begin_date)
        stockList = [stock for stock in stockList if not stock.startswith(('3', '68', '4', '8'))]
    
    st_data = get_extras('is_st', stockList, count=1, end_date=begin_date)
    stockList = [stock for stock in stockList if not st_data[stock][0]]
    stockList = delect_stop(stockList, begin_date)
    return stockList

def get_trade_dates(start_date, end_date):
    """获取交易日列表"""
    return get_trade_days(start_date=start_date, end_date=end_date)

def get_factor_data(securities_list, jqfactors_list, date):
    """获取指定日期的因子数据"""
    factor_data = get_factor_values(securities=securities_list, factors=jqfactors_list, count=1, end_date=date)
    df_jq_factor = pd.DataFrame(index=securities_list)
    for i in factor_data.keys():
        df_jq_factor[i] = factor_data[i].iloc[0, :]
    return df_jq_factor

def get_factor_data_batch(securities_list, jqfactors_list, date, batch_size=500):
    """分批获取指定日期的因子数据"""
    all_factor_data = []
    
    # 分批处理股票列表
    for i in range(0, len(securities_list), batch_size):
        batch_securities = securities_list[i:i + batch_size]
        
        try:
            df_batch = get_factor_data(batch_securities, jqfactors_list, date)
            all_factor_data.append(df_batch)
            print(f"已获取第 {i//batch_size + 1} 批因子数据，股票数量: {len(batch_securities)}")
            
        except Exception as e:
            print(f"获取第 {i//batch_size + 1} 批因子数据时出错: {e}")
            continue
    
    # 合并所有批次的数据
    if all_factor_data:
        return pd.concat(all_factor_data, axis=0)
    else:
        return pd.DataFrame()

def get_price_data(securities_list, date):
    """获取指定日期的价格数据"""
    try:
        price_data = get_price(
            securities_list, 
            end_date=date, 
            frequency='daily', 
            fields=['open', 'close', 'low', 'high', 'volume'], 
            panel=False, 
            count=1
        )
        if price_data.empty:
            return None
        # 重新格式化数据 - 修复：正确处理DataFrame格式
        price_dict = {}
        for _, row in price_data.iterrows():
            stock = row['code']
            price_dict[stock] = {
                'open': row['open'],
                'close': row['close'],
                'low': row['low'],
                'high': row['high'],
                'volume': row['volume']
            }
        
        return price_dict
    except Exception as e:
        print(f"获取价格数据时出错: {e}")
        return None

def get_valuation_data(securities_list, date):
    """获取指定日期的市值数据"""
    try:
        q = query(
            valuation.code,
            valuation.market_cap,
            valuation.circulating_market_cap
        ).filter(
            valuation.code.in_(securities_list)
        )
        valuation_data = get_fundamentals(q, date=date)
        
        if valuation_data.empty:
            return None
        
        # 转换为字典格式便于查找
        valuation_dict = {}
        for _, row in valuation_data.iterrows():
            stock = row['code']
            valuation_dict[stock] = {
                'market_cap': row['market_cap'],
                'circulating_market_cap': row['circulating_market_cap']
            }
        
        return valuation_dict
    except Exception as e:
        print(f"获取市值数据时出错: {e}")
        return None

def save_to_hdf5(data_list, filename, mode='a'):
    """保存数据到HDF5文件"""
    if not data_list:
        return
    
    df = pd.concat(data_list, ignore_index=True)
    df.to_hdf(filename, key='data', mode=mode, format='table', append=(mode=='a'))
    print(f"已保存 {len(df)} 条记录到 {filename}")

def filter_factors_by_nan_ratio(securities_list, factors_list, date_str, nan_threshold=0.5):
    """根据NaN值比例过滤因子"""
    try:
        print(f"开始过滤日期 {date_str} 的因子，NaN阈值: {nan_threshold}")
        
        # 获取因子数据
        factor_data = get_factor_data_batch(securities_list, factors_list, date_str, batch_size=500)
        
        if factor_data.empty:
            print(f"日期 {date_str} 因子数据为空")
            return []
        
        # 计算每个因子的NaN比例并过滤
        valid_factors = []
        for factor in factors_list:
            if factor in factor_data.columns:
                nan_ratio = factor_data[factor].isna().sum() / len(factor_data)
                if nan_ratio <= nan_threshold:
                    valid_factors.append(factor)
                else:
                    print(f"因子 {factor} NaN比例为 {nan_ratio:.2%}，已过滤")
            else:
                print(f"因子 {factor} 在数据中不存在，已过滤")
        
        print(f"日期 {date_str}: 原始因子数量 {len(factors_list)}，过滤后 {len(valid_factors)}")
        return valid_factors
        
    except Exception as e:
        print(f"过滤因子时出错: {e}")
        return factors_list

def load_progress():
    """加载进度文件"""
    if os.path.exists(PROGRESS_FILE):
        try:
            progress_df = pd.read_csv(PROGRESS_FILE)
            if not progress_df.empty:
                start_date = progress_df['start_date'].iloc[0]
                end_date = progress_df['end_date'].iloc[0]
                print(f"从进度文件加载: 已处理日期范围 {start_date} 到 {end_date}")
                return start_date, end_date
        except Exception as e:
            print(f"加载进度文件失败: {e}")
    return None, None

def save_progress(start_date, end_date):
    """保存进度到文件"""
    try:
        progress_df = pd.DataFrame({'start_date': [start_date], 'end_date': [end_date]})
        progress_df.to_csv(PROGRESS_FILE, index=False)
    except Exception as e:
        print(f"保存进度文件失败: {e}")

def update_progress(current_start, current_end, new_date):
    """更新进度并保存"""
    if current_start is None:
        start_date = new_date
    else:
        start_date = min(current_start, new_date)
    
    if current_end is None:
        end_date = new_date
    else:
        end_date = max(current_end, new_date)
    
    save_progress(start_date, end_date)
    return start_date, end_date

# %% 获取所有因子列表
print("获取所有因子列表...")
all_factors_df = get_all_factors()
jqfactors_list = all_factors_df['factor'].tolist()
print(f"总共获取到 {len(jqfactors_list)} 个因子")

# %% 主要数据获取流程
print("开始获取因子数据...")

# 获取交易日列表
trade_dates = get_trade_dates(START_DATE, END_DATE)
trade_dates = [date.date() if hasattr(date, 'date') else date for date in trade_dates]
print(f"交易日数量: {len(trade_dates)}")

# 加载进度
progress_start, progress_end = load_progress()
is_resume = progress_start is not None

# 初始化数据存储
data_batch = []
total_processed = 0

# 删除已存在的文件（仅在非恢复模式下）
if not is_resume and os.path.exists(DATA_FILE):
    os.remove(DATA_FILE)
    print(f"已删除现有文件: {DATA_FILE}")

# 遍历每个交易日
for i, date in enumerate(tqdm(trade_dates, desc="处理交易日")):
    try:
        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
        
        # 跳过已处理的日期范围
        if is_resume and progress_start and progress_end:
            if progress_start <= date_str <= progress_end:
                continue
        
        stockList = get_stock(STOCK_POOL, date_str)
        if stockList is None or len(stockList) == 0:
            continue

        # 获取价格数据
        price_data = get_price_data(stockList, date_str)
        if price_data is None:
            print(f'日期 {date_str} 价格数据为空')
            continue
        
        # 获取市值数据
        valuation_data = get_valuation_data(stockList, date_str)
        if valuation_data is None:
            print(f'日期 {date_str} 市值数据为空')
            continue
        
        # 合并价格和市值数据
        final_data = []
        for stock in stockList:
            if stock in price_data and stock in valuation_data:
                row_data = {
                    'date': date_str,
                    'code': stock,
                    'open': price_data[stock]['open'],
                    'close': price_data[stock]['close'], 
                    'low': price_data[stock]['low'],
                    'high': price_data[stock]['high'],
                    'volume': price_data[stock]['volume'],
                    'market_cap': valuation_data[stock]['market_cap'],
                    'circulating_market_cap': valuation_data[stock]['circulating_market_cap']
                }
                
                final_data.append(row_data)
        if final_data:
            df_final = pd.DataFrame(final_data)
            data_batch.append(df_final)
            total_processed += len(df_final)
        # 每隔SAVE_BATCH_SIZE次或最后一次保存数据
        if (i + 1) % SAVE_BATCH_SIZE == 0 or i == len(trade_dates) - 1:
            if data_batch:
                save_to_hdf5(data_batch, DATA_FILE, mode='a')
                # 更新进度范围
                batch_dates = []
                for batch_data in data_batch:
                    batch_dates.extend(batch_data['date'].unique())
                if batch_dates:
                    min_date = min(batch_dates)
                    max_date = max(batch_dates)
                    progress_start, progress_end = update_progress(progress_start, progress_end, min_date)
                    progress_start, progress_end = update_progress(progress_start, progress_end, max_date)
                data_batch = []  # 清空批次数据
                gc.collect()  # 强制垃圾回收
                print(f"已处理 {i+1}/{len(trade_dates)} 个交易日，累计记录: {total_processed}")
    
    except Exception as e:
        print(f"处理日期 {date} 时出错: {e}")
        continue

print(f"\n数据获取完成！")
print(f"总共处理了 {total_processed} 条记录")
print(f"数据已保存到: {DATA_FILE}")

# %% 获取指数自身量价数据
INDEX_CODE = '399101.XSHE'  # 中小板指数代码

print(f"\n开始获取指数 {INDEX_CODE} 自身量价数据...")

try:
    # 获取指数价格数据
    index_price_data = get_price(
        INDEX_CODE,
        start_date=START_DATE,
        end_date=END_DATE,
        frequency='daily',
        fields=['open', 'close', 'low', 'high', 'volume'],
        panel=False
    )
    
    if not index_price_data.empty:
        # 格式化指数数据
        index_data = []
        for date_idx, row in index_price_data.iterrows():
            index_row = {
                'date': date_idx.strftime('%Y-%m-%d'),
                'code': INDEX_CODE,
                'open': row['open'],
                'close': row['close'],
                'low': row['low'],
                'high': row['high'],
                'volume': row['volume'],
                'market_cap': np.nan,  # 使用NaN保持数据类型一致
                'circulating_market_cap': np.nan
            }
            index_data.append(index_row)
        
        # 转换为DataFrame并保存
        index_df = pd.DataFrame(index_data)
        save_to_hdf5([index_df], DATA_FILE, mode='a')
        
        print(f"指数 {INDEX_CODE} 数据获取完成，共 {len(index_df)} 条记录")
    else:
        print(f"指数 {INDEX_CODE} 数据为空")

except Exception as e:
    print(f"获取指数数据时出错: {e}")

print(f"\n所有数据获取完成！数据文件: {DATA_FILE}")
