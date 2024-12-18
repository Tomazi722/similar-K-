"""相似K线代码使用教程 """

"""一、可改代码"""

# 确认需要验证的相似K线所处的时间段
start_time = '1990/01/01'
end_time = '2024/10/17'

# 确认需要验证哪个股票
stock_name = 'sh000001'

# 确认想要验证的K线自身在哪段时间
# stock_start_time 为具体时间时，即为该时间段的K线
# stock_start_time 为 None 时，则为 stock_end_time 向前最近20个交易日的K线
# length 指定看多少个交易日的K线
length = 20
stock_start_time = None
stock_end_time = '2024/10/17'

# 确认在自己历史上找相似K线还是在所有股票找？
# False 是在所有股票，True是在自己历史上
only_self = False

# 选择读取的数据文件夹
stock_path = r'index/'
# stock_path = r'股票历史日线数据/'

# 最多会选多少组相似K线做评价
max_num = 10000

# 选择大于该相似度的相似K线做评价
least_r = 0.9

# 未来1/2/3/5/10/20天的表现
day_list = [1, 2, 3, 5, 10, 20]

# 输出图片中，向后画多少天的表现
future_days = 5

# 输出图片中，画多少组K线
future_pics = 5



"""二、实现对个股自身历史相似K线的统计"""
# 核心代码
only_self = True

# 1、找最新的20根K线
length = 20
stock_start_time = None
stock_end_time = '2024/10/17'

# 2、找指定时间段的K线
stock_start_time = '2024/09/17'
stock_end_time = '2024/10/17'



"""三、实现在所有股票中寻找相似K线的统计"""
# 核心代码
only_self = False

# 1、找最新的20根K线
length = 20
stock_start_time = None
stock_end_time = '2024/10/17'

# 2、找指定时间段的K线
stock_start_time = '2024/09/17'
stock_end_time = '2024/10/17'

# 3、选择指定数量的最相似K线做评价
max_num = 10000

# 4、选择指定相似度以上的相似K线做评价
least_r = 0.9



"""四、其他问题"""
# 其他问题可在微信咨询邢不行