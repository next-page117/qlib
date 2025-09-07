import os
import logging
import time


class BacktestTimeFormatter(logging.Formatter):
    """自定义日志格式器，使用回测时间戳"""
    
    def __init__(self, fmt=None, datefmt=None, strategy=None):
        super().__init__(fmt, datefmt)
        self.strategy = strategy
        
    def formatTime(self, record, datefmt=None):
        """重写formatTime方法，使用回测时间戳"""
        if self.strategy and hasattr(self.strategy, 'trade_calendar'):
            try:
                # 获取当前交易步数和对应的时间
                trade_step = self.strategy.trade_calendar.get_trade_step()
                trade_start_time, trade_end_time = self.strategy.trade_calendar.get_step_time(trade_step)
                # 使用交易开始时间作为日志时间戳
                return trade_start_time.strftime('%Y-%m-%d %H:%M:%S')
            except:
                # 如果获取回测时间失败，回退到默认行为
                pass
        
        # 默认行为：使用实际时间
        if datefmt:
            return time.strftime(datefmt, self.converter(record.created))
        else:
            return time.strftime('%Y-%m-%d %H:%M:%S', self.converter(record.created))


def setup_backtest_logger(logger_name, log_filename, strategy_instance=None):
    """
    配置回测专用日志器
    
    Parameters
    ----------
    logger_name : str
        日志器名称
    log_filename : str
        日志文件名
    strategy_instance : object, optional
        策略实例，用于获取回测时间
        
    Returns
    -------
    logging.Logger
        配置好的日志器实例
    """
    # 获取当前文件所在目录的父目录，即策略项目根目录
    current_dir = os.path.dirname(os.path.dirname(__file__))
    log_dir = os.path.join(current_dir, "log")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, log_filename)
    
    # 强制删除已存在的日志文件
    if os.path.exists(log_file):
        os.remove(log_file)
    
    logger = logging.getLogger(logger_name)
    
    # 清除所有已存在的处理器
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    
    # 重新配置日志
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    
    # 使用自定义格式器，传入策略实例以获取回测时间
    formatter = BacktestTimeFormatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        strategy=strategy_instance
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    
    return logger
