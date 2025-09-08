# %% 包导入
import qlib
import pandas as pd
import sys
from pathlib import Path
from qlib.constant import REG_CN
from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import flatten_dict

# 添加自定义模块路径
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir))

# 导入自定义类
from dataset.alpha158_5day import Alpha158_5Day

# %% 内存清理
vars_to_drop = ["model","dataset","pred_df","report_normal_df","positions","analysis_df","pred_label"]
for v in vars_to_drop:
    if v in globals(): del globals()[v]
import gc; gc.collect()
pd.set_option('display.max_columns', None)

# %%
provider_uri = "~/.qlib/qlib_data/jukuan_data"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)

# %%
market = "399101" # 中小综指

# %% 模型训练
data_handler_config = {
    "start_time": "2010-01-01",
    "end_time": "2025-09-04",
    "fit_start_time": "2010-01-01",
    "fit_end_time": "2020-12-31",
    "instruments": market,
}

task = {
    "model": {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "mse",
            "colsample_bytree": 0.8879,
            "learning_rate": 0.0421,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": 8,
            "num_leaves": 210,
            "num_threads": 20,
        },
    },
    "dataset": {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158_5Day",  # 使用自定义数据处理类
                "module_path": "dataset.alpha158_5day",  # 自定义模块路径
                "kwargs": data_handler_config,
            },
            "segments": {
                "train": ("2010-01-01", "2020-12-31"),
                "valid": ("2021-01-01", "2022-12-31"),
                "test": ("2023-01-01", "2025-09-04"),
            },
        },
    },
}

# 直接初始化自定义Handler和Dataset
handler = Alpha158_5Day(**data_handler_config)
# 手动创建DatasetH实例
from qlib.data.dataset import DatasetH
dataset = DatasetH(
    handler=handler,
    segments={
        "train": ("2010-01-01", "2020-12-31"),
        "valid": ("2021-01-01", "2022-12-31"),
        "test": ("2023-01-01", "2025-09-04"),
    }
)

# ================= 手工训练 =================
model = init_instance_by_config(task["model"])

# 训练（可下断点）
train_x = dataset.prepare("train", col_set="feature")
train_y = dataset.prepare("train", col_set="label")
model.fit(dataset) 

# 验证/测试集特征与标签（调试查看）
valid_x = dataset.prepare("valid", col_set="feature")
valid_y = dataset.prepare("valid", col_set="label")
test_x  = dataset.prepare("test",  col_set="feature")
test_y  = dataset.prepare("test",  col_set="label")

# %% 预测
pred = model.predict(dataset, segment="test")
pred.name = "pred"
# 组合 label+pred
pred_df = pd.concat([test_y.rename(columns={test_y.columns[0]: "label"}), pred], axis=1)
print("预测结果示例：")
print(pred_df.head())

# 保存预测结果到cache
cache_dir = current_dir / "cache"
cache_dir.mkdir(exist_ok=True)
pred_cache_path = cache_dir / "pred_results.pkl"
pred.to_pickle(pred_cache_path)
print(f"预测结果已保存到: {pred_cache_path}")

# %% 回测 - 使用自定义策略
from strategy.top5_5day_strategy import Top5_5DayStrategy
from qlib.backtest.executor import SimulatorExecutor
from qlib.backtest import backtest
from qlib.contrib.evaluate import risk_analysis

# 从cache读取预测结果
pred_cached = pd.read_pickle(pred_cache_path)

# 使用缓存的预测结果
strategy = Top5_5DayStrategy(
    signal=pred_cached.to_frame("score") if hasattr(pred_cached, 'to_frame') else pred_cached,
    topk=5,           # 选择5只股票
    trade_period=5,   # 5天调仓一次
)
executor = SimulatorExecutor(
    time_per_step="day",
    generate_portfolio_metrics=True,
)

benchmark = "399101.xshe"
backtest_start = "2023-01-01"
backtest_end   = "2025-09-04"   # 注意不要用最后一个交易日 (需要下一日价格计算收益)

portfolio_metric_dict, indicator_dict = backtest(
    start_time=backtest_start,
    end_time=backtest_end,
    strategy=strategy,
    executor=executor,
    benchmark=benchmark,
    account=100000000,
    exchange_kwargs={
        "freq": "day",
        "limit_threshold": 0.095,
        "deal_price": "close",
        "open_cost": 0.0005,
        "close_cost": 0.0015,
        "min_cost": 5,
    },
)

# 取出 1day 频率的报告与持仓
report_normal_df, positions = portfolio_metric_dict["1day"]
excess_no_cost = report_normal_df["return"] - report_normal_df["bench"]
excess_with_cost = report_normal_df["return"] - report_normal_df["bench"] - report_normal_df["cost"]

freq = "1day"  # 与生成文件名一致
ana_no_cost = risk_analysis(excess_no_cost, freq=freq)
ana_with_cost = risk_analysis(excess_with_cost, freq=freq)

analysis_df = pd.concat(
    {
        "excess_return_without_cost": ana_no_cost,
        "excess_return_with_cost": ana_with_cost,
    }
)
print("回测分析结果：")
print(analysis_df)

# %% 绘图
from qlib.contrib.report.analysis_position.parse_position import parse_position
from qlib.contrib.report import analysis_position, analysis_model

# 收益图
analysis_position.report_graph(report_normal_df)

# 风险分析
analysis_position.risk_analysis_graph(analysis_df, report_normal_df)

# IC（需要对齐 index）
pred_only = pred_cached.to_frame("score")
test_label_df = test_y.rename(columns={test_y.columns[0]: "label"})
pred_label = pd.concat([test_label_df, pred_only], axis=1).reindex(test_label_df.index)
analysis_position.score_ic_graph(pred_label)

# 模型表现
analysis_model.model_performance_graph(pred_label)

