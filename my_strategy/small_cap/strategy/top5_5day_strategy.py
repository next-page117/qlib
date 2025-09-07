import pandas as pd
import numpy as np
from qlib.contrib.strategy.signal_strategy import BaseSignalStrategy
from qlib.strategy.base import BaseStrategy
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from qlib.backtest.position import Position


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
        
    def generate_trade_decision(self, execute_result=None):
        """生成交易决策"""
        # 获取当前交易步数和时间
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        
        # 判断是否需要调仓
        if not self._should_trade(trade_step):
            return TradeDecisionWO([], self)
            
        # 获取信号 - 使用前一期的信号预测当前期
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        
        # 处理多列信号的情况
        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]
            
        if pred_score is None or pred_score.empty:
            return TradeDecisionWO([], self)
            
        # 按分数排序，选择前topk只股票
        pred_score = pred_score.dropna().sort_values(ascending=False)
        target_stocks = pred_score.head(self.topk).index.tolist()
        
        # 获取当前持仓
        current_position = self.trade_position
        current_stocks = set(current_position.get_stock_list())
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
                    continue
                    
                # 全部卖出
                current_amount = current_position.get_stock_amount(stock)
                if current_amount > 0:
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
        
        # 买入目标股票
        for stock in target_stocks:
            # 检查股票是否可交易
            if not self.trade_exchange.is_stock_tradable(
                stock_id=stock,
                start_time=trade_start_time, 
                end_time=trade_end_time,
                direction=OrderDir.BUY
            ):
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
        
        # 更新最后交易步数
        self.last_trade_step = trade_step
        
        return TradeDecisionWO(order_list, self)
    
    def _should_trade(self, current_step):
        """判断是否应该交易"""
        if self.last_trade_step is None:
            return True
            
        # 计算距离上次交易的步数
        steps_diff = current_step - self.last_trade_step
        return steps_diff >= self.trade_period
