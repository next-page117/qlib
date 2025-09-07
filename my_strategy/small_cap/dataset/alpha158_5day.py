import pandas as pd
from qlib.contrib.data.handler import Alpha158
from qlib.data import D


class Alpha158_5Day(Alpha158):
    """
    继承Alpha158，增加五日后收益率作为label
    """
    
    def get_label_config(self):
        """重写label配置，使用5日后收益率"""
        return ["Ref($close, -5)/$close - 1"], ["LABEL0"]
    
    def fetch_data(self):
        """获取数据，包括特征和标签"""
        # 获取原始特征数据
        feature_data = super().fetch_data()
        
        # 获取标签配置
        label_config, label_names = self.get_label_config()
        
        # 获取标签数据
        label_data = D.features(
            self.instruments, 
            label_config,
            self.start_time,
            self.end_time,
            freq=self.freq,
            inst_processors=self.inst_processors,
        )
        label_data.columns = label_names
        
        # 合并特征和标签
        if feature_data is not None and not feature_data.empty:
            data = pd.concat([feature_data, label_data], axis=1, sort=True)
        else:
            data = label_data
            
        return data
