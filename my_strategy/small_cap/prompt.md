# 目标
完成自定义的训练和回测流程。
# 详细说明
## 数据处理
1. 将默认的alpha158label中增加五日后收益率的label，使用继承类的方式实现。
## 回测
1. 也是生成自定义的回测类，回测的调仓周期为五天，选择预测分数最高的五只股票。
# 编码说明
1. 数据处理自定义的dataset放到my_strategy/small_cap/dataset文件夹中，名称自定义。
2. 回测自定义类放到my_strategy/small_cap/strategy文件夹将中，名称自定义。
3. 主要执行文件还是my_strategy/small_cap/workflow/v2.py。
4. 不要修改qlib的底层源码，使用自定义类实现，通过v2.py的配置选项选择自定义类。
5. 除了上述描述的文件，不要修改其它文件，代码实现尽量简单。
