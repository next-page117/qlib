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