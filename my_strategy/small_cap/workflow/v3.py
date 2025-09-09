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

# 保存结果为csv文件
pred_csv_df = (
    pred.rename("score")
        .reset_index()  # MultiIndex -> 普通列
        .rename(columns={"datetime": "date", "instrument": "code"})
)
csv_path = cache_dir / "pred_results.csv"
pred_csv_df.to_csv(csv_path, index=False, encoding="utf-8-sig", float_format="%.6f")
print(f"预测结果CSV已保存: {csv_path}  行数={len(pred_csv_df)}  样例：")
print(pred_csv_df.head())

# %% 模型性能分析
import sys
from pathlib import Path
current_dir = Path(__file__).parent.parent if hasattr(Path(__file__), 'parent') else Path.cwd().parent
sys.path.insert(0, str(current_dir))
from qlib.constant import REG_CN
import pandas as pd
from qlib.contrib.report.analysis_position.parse_position import parse_position
from qlib.contrib.report import analysis_position, analysis_model

# 从cache读取预测结果
pred_cached = pd.read_pickle("../cache/pred_results.pkl")
pred_only = pred_cached.to_frame("score")
test_label_df = test_y.rename(columns={test_y.columns[0]: "label"})
pred_label = pd.concat([test_label_df, pred_only], axis=1).reindex(test_label_df.index)
analysis_position.score_ic_graph(pred_label)

# 模型表现
analysis_model.model_performance_graph(pred_label)

