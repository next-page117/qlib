import pandas as pd
import numpy as np
from qlib.contrib.strategy.signal_strategy import BaseSignalStrategy
from qlib.strategy.base import BaseStrategy
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from qlib.backtest.position import Position
from qlib.data import D
import sys
from pathlib import Path
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))
from log.backtest_logger import setup_backtest_logger

def get_features(instruments, fields, start_time, end_time):
    """获取股票特征数据，返回包含股票代码、时间、查询字段的DataFrame"""
    features_df = D.features(instruments, fields, start_time, end_time).reset_index()
    for field in fields:
        if field in features_df.columns:
            # 彻底的类型转换：先转数值，再去除无效值，最后确保为float
            features_df[field] = pd.to_numeric(features_df[field], errors='coerce')
            features_df[field] = features_df[field].replace([np.inf, -np.inf], np.nan)
            features_df[field] = features_df[field].astype('float64')
    return features_df

class Top5_5DayStrategy(BaseSignalStrategy):
    """
    自定义策略：
    - 五天调仓一次
    - 每次选择预测分数最高的5只股票
    - 等权重持仓
    """
    
    def __init__(self, signal, topk=5, trade_period=5, **kwargs):
        """
        Parameters
        ----------
        signal : pd.DataFrame or BaseModel
            预测信号
        topk : int
            选择股票数量，默认5只
        trade_period : int  
            调仓周期（天），默认5天
        """
        super().__init__(signal=signal, **kwargs)
        self.topk = topk
        self.trade_period = trade_period
        self.last_trade_step = None
        
        # 使用自定义日志配置
        self.logger = setup_backtest_logger(
            logger_name="Top5_5DayStrategy",
            log_filename="top5_5day_strategy.log",
            strategy_instance=self
        )
        self.logger.info(f"策略初始化: topk={topk}, trade_period={trade_period}")
        
    def _log_current_holdings(self, trade_step, current_position, pred_score):
        """打印当前持仓信息"""
        current_stocks = current_position.get_stock_list()
        if not current_stocks:
            self.logger.info(f"Step {trade_step}: 当前无持仓")
            return
            
        self.logger.info(f"Step {trade_step}: 当前持仓详情:")
        for stock in current_stocks:
            amount = current_position.get_stock_amount(stock)
            price = current_position.get_stock_price(stock)
            value = amount * price
            score = pred_score[stock] if stock in pred_score.index else None
            self.logger.info(f"  {stock}: 数量={amount}, 价格={round(price, 2)}, 价值={round(value, 2)}, 预测得分={round(score, 4) if score is not None else 'N/A'}")
        
    def _select_target_stocks(self, pred_start_time, pred_end_time, trade_step):
        """
        选股函数：
        1. 筛选预测得分>0的股票
        2. 从中选择流通市值最小的topk只股票
        
        Returns
        -------
        list
            目标股票列表
        """
        # 获取预测信号
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        
        # 处理多列信号的情况
        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]
        
        if pred_score is None or pred_score.empty:
            self.logger.warning(f"Step {trade_step}: 无有效预测信号")
            return []
        
        # 步骤1：筛选预测得分>0的股票
        pred_score = pred_score.dropna()
        filtered_stocks = pred_score[pred_score > 0]
        
        if filtered_stocks.empty:
            self.logger.warning(f"Step {trade_step}: 无预测得分>0的股票")
            return []
        
        # 步骤2：获取这些股票的流通市值
        stocks = filtered_stocks.index.tolist()
        features_df = get_features(stocks, ["$circulating_market_cap"], pred_end_time, pred_end_time)
        
        # 从重置索引后的DataFrame中提取数据
        market_cap_data = features_df[features_df['datetime'] == pred_end_time]
        market_cap_series = market_cap_data.set_index('instrument')["$circulating_market_cap"].dropna()

        # 按流通市值升序排列，选最小的topk
        target_stocks = market_cap_series.nsmallest(self.topk).index.tolist()
        
        # 记录选股结果
        top5_scores = [(stock, round(pred_score[stock], 4), round(market_cap_series[stock], 2)) for stock in target_stocks]
        self.logger.info(f"Step {trade_step}: TOP{self.topk}选股: {top5_scores}")
        
        return target_stocks
        
    def generate_trade_decision(self, execute_result=None):
        """生成交易决策"""
        # 获取当前交易步数和时间
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        
        # 判断是否需要调仓
        if not self._should_trade(trade_step):
            self.logger.info(f"Step {trade_step}: 跳过调仓 - 未到调仓周期 (上次调仓: {self.last_trade_step})")
            return TradeDecisionWO([], self)
            
        self.logger.info(f"Step {trade_step}: 开始调仓 - 交易时间: {trade_start_time} ~ {trade_end_time}")
            
        # 获取信号 - 使用前一期的信号预测当前期
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        
        # 使用选股函数获取目标股票
        target_stocks = self._select_target_stocks(pred_start_time, pred_end_time, trade_step)
        
        if not target_stocks:
            self.logger.warning(f"Step {trade_step}: 无目标股票")
            return TradeDecisionWO([], self)
        
        # 获取预测分数用于后续记录
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]
        pred_score = pred_score.dropna()
        
        # 获取当前持仓
        current_position = self.trade_position
        current_stocks = set(current_position.get_stock_list())
        
        # 记录当前持仓及其预测得分
        self._log_current_holdings(trade_step, current_position, pred_score)
        
        target_stocks_set = set(target_stocks)
        
        # 生成订单列表
        order_list = []
        
        # 卖出不在目标股票中的持仓
        for stock in current_stocks:
            if stock not in target_stocks_set:
                # 检查股票是否可交易
                if not self.trade_exchange.is_stock_tradable(
                    stock_id=stock, 
                    start_time=trade_start_time, 
                    end_time=trade_end_time,
                    direction=OrderDir.SELL
                ):
                    self.logger.warning(f"Step {trade_step}: {stock} 不可卖出")
                    continue
                    
                # 全部卖出
                current_amount = current_position.get_stock_amount(stock)
                if current_amount > 0:
                    sell_price = self.trade_exchange.get_deal_price(
                        stock_id=stock,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                        direction=OrderDir.SELL
                    )
                    
                    order = Order(
                        stock_id=stock,
                        amount=current_amount,
                        direction=Order.SELL,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                    )
                    # 检查订单是否可执行
                    if self.trade_exchange.check_order(order):
                        order_list.append(order)
                        pred_score_val = pred_score[stock] if stock in pred_score.index else None
                        self.logger.info(f"Step {trade_step}: 卖出 {stock} - 数量: {current_amount}, 预估价格: {round(sell_price, 2)}, 预测得分: {round(pred_score_val, 4) if pred_score_val else None}")
        
        # 计算可用资金
        cash = current_position.get_cash()
        # 模拟执行卖出订单以更新现金
        import copy
        temp_position = copy.deepcopy(current_position)
        for order in order_list:
            if order.direction == Order.SELL:
                trade_val, trade_cost, _ = self.trade_exchange.deal_order(order, position=temp_position)
                cash += trade_val - trade_cost
        
        # 计算每只股票的目标金额
        target_value_per_stock = cash * self.get_risk_degree() / len(target_stocks) if target_stocks else 0
        self.logger.info(f"Step {trade_step}: 可用资金: {round(cash, 2)}, 单只股票目标金额: {round(target_value_per_stock, 2)}")
        
        # 买入目标股票
        for stock in target_stocks:
            # 检查股票是否可交易
            if not self.trade_exchange.is_stock_tradable(
                stock_id=stock,
                start_time=trade_start_time, 
                end_time=trade_end_time,
                direction=OrderDir.BUY
            ):
                self.logger.warning(f"Step {trade_step}: {stock} 不可买入")
                continue
                
            # 获取当前持仓价值（修复：手动计算股票价值）
            if current_position.check_stock(stock):
                current_amount = current_position.get_stock_amount(stock)
                current_price = current_position.get_stock_price(stock)
                current_value = current_amount * current_price
            else:
                current_value = 0
            
            # 计算需要调整的金额
            diff_value = target_value_per_stock - current_value
            
            if abs(diff_value) > 100:  # 避免微小调整
                # 获取买入价格
                buy_price = self.trade_exchange.get_deal_price(
                    stock_id=stock, 
                    start_time=trade_start_time, 
                    end_time=trade_end_time, 
                    direction=OrderDir.BUY
                )
                
                if buy_price > 0 and diff_value > 0:
                    # 计算买入数量
                    buy_amount = diff_value / buy_price
                    
                    # 按交易单位调整
                    factor = self.trade_exchange.get_factor(
                        stock_id=stock, 
                        start_time=trade_start_time, 
                        end_time=trade_end_time
                    )
                    buy_amount = self.trade_exchange.round_amount_by_trade_unit(buy_amount, factor)
                    
                    if buy_amount > 0:
                        order = Order(
                            stock_id=stock,
                            amount=buy_amount,
                            direction=Order.BUY,
                            start_time=trade_start_time,
                            end_time=trade_end_time,
                        )
                        order_list.append(order)
                        pred_score_val = pred_score[stock]
                        self.logger.info(f"Step {trade_step}: 买入 {stock} - 数量: {buy_amount}, 价格: {round(buy_price, 2)}, 当前价值: {round(current_value, 2)}, 目标价值: {round(target_value_per_stock, 2)}, 预测得分: {round(pred_score_val, 4)}")
            else:
                self.logger.info(f"Step {trade_step}: {stock} 调整金额过小({round(diff_value, 2)})，跳过")
        
        # 更新最后交易步数
        self.last_trade_step = trade_step
        
        self.logger.info(f"Step {trade_step}: 调仓完成 - 生成订单数: {len(order_list)}")
        
        return TradeDecisionWO(order_list, self)
    
    def _should_trade(self, current_step):
        """判断是否应该交易"""
        if self.last_trade_step is None:
            return True
            
        # 计算距离上次交易的步数
        steps_diff = current_step - self.last_trade_step
        return steps_diff >= self.trade_period
