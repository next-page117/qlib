# %%
import qlib
import pandas as pd
from qlib.constant import REG_CN
from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import flatten_dict

# %% 内存清理
vars_to_drop = ["model","dataset","pred_df","report_normal_df","positions","analysis_df","pred_label"]
for v in vars_to_drop:
    if v in globals(): del globals()[v]
import gc; gc.collect()

# %%
provider_uri = "~/.qlib/qlib_data/jukuan_data"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)

# %%
market = "399101" # 中小综指
benchmark = "399101.xshe"

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
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
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

# model initialization
model = init_instance_by_config(task["model"])
dataset = init_instance_by_config(task["dataset"])

# start exp to train model
with R.start(experiment_name="train_model"):
    R.log_params(**flatten_dict(task))
    model.fit(dataset)
    R.save_objects(trained_model=model)
    rid = R.get_recorder().id

# %% 预测回测分析
port_analysis_config = {
    "executor": {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
        },
    },
    "strategy": {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy.signal_strategy",
        "kwargs": {
            "model": model,
            "dataset": dataset,
            "topk": 50,
            "n_drop": 5,
        },
    },
    "backtest": {
        "start_time": "2023-01-01",
        "end_time": "2025-09-04",
        "account": 100000000,
        "benchmark": benchmark,
        "exchange_kwargs": {
            "freq": "day",
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
    },
}

# backtest and analysis
with R.start(experiment_name="backtest_analysis"):
    recorder = R.get_recorder(recorder_id=rid, experiment_name="train_model")
    model = recorder.load_object("trained_model")

    # prediction
    recorder = R.get_recorder()
    ba_rid = recorder.id
    sr = SignalRecord(model, dataset, recorder)
    sr.generate()

    # backtest & analysis
    par = PortAnaRecord(recorder, port_analysis_config, "day")
    par.generate()

# %% 分析绘图
from qlib.contrib.report import analysis_model, analysis_position
from qlib.data import D

recorder = R.get_recorder(recorder_id=ba_rid, experiment_name="backtest_analysis")
print(recorder)
pred_df = recorder.load_object("pred.pkl")
report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")

# %%
analysis_position.report_graph(report_normal_df)

# %%
analysis_position.risk_analysis_graph(analysis_df, report_normal_df)

# %% 持仓分析，会报错
# pred_df_dates = pred_df.index.get_level_values(level='datetime')
# features_df = D.features(D.instruments(market), ['Ref($close, -1)/$close-1'], pred_df_dates.min(), pred_df_dates.max())
# features_df.columns = ['label']
# analysis_position.rank_label_graph(positions, features_df, pred_df_dates.min(), pred_df_dates.max())

# %%
label_df = dataset.prepare("test", col_set="label")
label_df.columns = ["label"]

# %%
pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)
analysis_position.score_ic_graph(pred_label)

# %%
analysis_model.model_performance_graph(pred_label)
