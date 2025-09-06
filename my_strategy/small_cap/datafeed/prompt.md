# 目标
增加市值数据、流通市值数据到最终数据中。
# 详细说明
1. 市值数据和流通数据获取实例
```python
    date = "2025-09-05"
    stock_list = get_index_stocks('399101.XSHE', date)
    q = query(
            valuation.code,
            valuation.market_cap,
            valuation.circulating_market_cap
            ).filter(
                valuation.code.in_(stock_list)
            )
    get_fundamentals(q, date=date)
```
数据结构示例：
    code	market_cap	circulating_market_cap
0	002001.XSHE	739.4653	730.6690
1	002003.XSHE	123.6685	107.5578

# 编码要求
1. 改动可能少，代码需要简洁

# 目标
将聚宽获取的数据转换为qlib能识别的bin文件。
# 详细说明
## 原始数据说明
1. 聚宽下载的文件为h5格式，包含以下关键列：
'date': date_str,
'code': stock,
'open': price_data[stock]['open'],
'close': price_data[stock]['close'], 
'low': price_data[stock]['low'],
'high': price_data[stock]['high'],
'volume': price_data[stock]['volume'],
'market_cap': valuation_data[stock]['market_cap'],
'circulating_market_cap': valuation_data[stock]['circulating_market_cap']
需要注意的是，index为数字，时间在date列中。
2. 聚宽原始文件路径为/mnt/c/nero/quant/data/jukuan/small_cap_olhcv_data.h5
## 转换说明
1. 转换流程参考scripts/dump_bin.py文件，该文件的输入为多个标的的csv文件，因此你需要对该文件进行改造以适配现有的数据格式。
2. 转换后的数据文件路径为/home/nero/.qlib/qlib_data/jukuan_data
# 编码说明
1. 代码尽量简洁，不要写的过于复杂。
2. 代码放到my_strategy/small_cap/datafeed/prompt.md，不要修改其它文件。