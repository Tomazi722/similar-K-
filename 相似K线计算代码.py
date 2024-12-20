import akshare as ak
from similarity import *
import warnings
import pandas as pd
import os

def get_stock_data(stock_code, start_date, end_date):
    """
    从akshare获取股票数据
    :param stock_code: 股票代码 (如 'sh000001')
    :param start_date: 开始日期
    :param end_date: 结束日期
    :return: DataFrame
    """
    try:
        # 对于上证指数，需要特殊处理
        if stock_code == 'sh000001':
            df = ak.stock_zh_index_daily(symbol="sh000001")
            # 对于指数，计算成交额（volume * close 作为估算）
            df['amount'] = df['volume'] * df['close']
        else:
            # 移除市场标识符（sh/sz）
            pure_code = stock_code[2:]
            df = ak.stock_zh_a_hist(symbol=pure_code, start_date=start_date.replace('/', ''), 
                                  end_date=end_date.replace('/', ''), adjust="qfq")
            # 个股数据中已经包含 amount 列，只需重命名
            df = df.rename(columns={'成交额': 'amount'})
        # print(df.tail(100))
        # 统一列名
        df['candle_end_time'] = pd.to_datetime(df['日期'])
        df = df.rename(columns={
            '开盘': 'open',
            '最高': 'high',
            '最低': 'low',
            '收盘': 'close',
            '成交量': 'volume'
        })
        
        # 选择需要的列
        df = df[['candle_end_time', 'open', 'high', 'low', 'close', 'volume', 'amount']]
        return df
    except Exception as e:
        print(f"获取{stock_code}数据时出错: {str(e)}")
        return None

def calculate_similarity_process(args):
    return calculate_similarity(*args)

def main():
    warnings.filterwarnings("ignore")
    pd.set_option('expand_frame_repr', False)
    pd.set_option('display.max_rows', 5000)

    # 确认需要验证的相似K线所处的时间段
    start_time = '1990/01/01'
    end_time = '2024/12/20'

    # 确认需要验证哪个股票
    stock_name = 'sz603220 '

    # 直接在当前目录存储数据
    base_path = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件所在目录
    print(f"当前文件所在目录: {base_path}")
    
    # 先获取数据并保存
    index_file = os.path.join(base_path, f'{stock_name}.csv')
    print(f"尝试读取文件: {index_file}")
    
    # 强制从AKShare获取新数据
    print(f"从AKShare获取{stock_name}数据...")
    index_df = get_stock_data(stock_name, start_time, end_time)
    if index_df is None:
        print("获取数据失败，程序退出")
        return
    
    # 添加 index_code 列
    index_df['index_code'] = stock_name
    
    # 保存数据到本地
    index_df.to_csv(index_file, encoding='gbk', index=False)
    print(f"数据已保存到 {index_file}")
    base_path = base_path+"\\"
    # 确认想要验证的K线自身在哪段时间
    # stock_start_time 为具体时间时，即为该时间段的K线
    # stock_start_time 为 None 时，则为 stock_end_time 向前最近20个交易日的K线
    # length 指定看多少个交易日的K线
    length = 20
    stock_end_time = '2024/10/17'
    stock_start_time = None

    # 最多会选多少组相似K线做评价
    max_num = 10000

    # 确认在自己历史上找相似K线还是在所有股票找？
    # False 是在所有股票，True是在自己历史上
    only_self = True

    # 选择大于该相似度的相似K线做评价
    least_r = 0.9

    # 未来1/2/3/5/10/20天的表现
    day_list = [1, 2, 3, 5, 10, 20]

    # 输出图片中，向后画多少天的表现
    future_days = 5
    # 输出图片中，画多少组K线
    future_pics = 5

    # ======================第一步，计算相似度
    params = length, start_time, end_time, stock_name, stock_start_time, stock_end_time, index_df, base_path, max_num, least_r, only_self
    df_result_r = calculate_similarity_process(params)
    index_df.rename(columns={'candle_end_time': '交易日期'}, inplace=True)
    
    # 修复结果文件路径
    result_file = os.path.join(base_path, 'D_0_相识度计算结果.csv')
    df_result_r.to_csv(result_file, index=False)
    print(f"相似度计算结果已保存到: {result_file}")

    # ======================第二步，准备画图数据
    dfs = []
    df_result_r_pic = df_result_r[df_result_r['r'].rank(ascending=False) <= future_pics+1]
    df_result_r_pic = df_result_r_pic.groupby('stock', as_index=False).agg({'startdate': list,'enddate': list,'r':list})
    
    for index, row in df_result_r_pic.iterrows():
        mini_df = cal_for_pic(row, file_path=base_path, future_days=future_days)
        if mini_df is not None:
            dfs.append(mini_df)
    
    if dfs:
        pic_df = pd.concat(dfs, ignore_index=True)
        pic_df = pic_df.sort_values(by=['r','股票代码', 'enddate'], ascending=[False,True, True])

    # ======================第三步，分析相似度文件
    dfs = []
    index_df['指数收盘价'] = index_df['close']
    index_df['指数开盘价'] = index_df['open']
    index_df = index_df[['交易日期', '指数收盘价', '指数开盘价']]
    df_result_r = df_result_r.groupby('stock',as_index=False).agg({'enddate': list})
    
    for index, row in df_result_r.iterrows():
        f = os.path.join(base_path, f"{row['stock']}.csv")
        enddate_list = row['enddate']
        mini_df = process_r_file(f, base_path, index_df, day_list, enddate_list)
        if mini_df is not None:
            dfs.append(mini_df)
    
    if dfs:
        all_df = pd.concat(dfs, ignore_index=True)
        analysis_df = analysis_for_r(all_df, day_list)
        print("\n分析结果:")
        print(analysis_df.T)

    # ======================第四步，画图
    if 'pic_df' in locals():
        draw_data(pic_df, future_pics, length, stock_start_time, stock_end_time, future_days, index_df)
    else:
        print("没有足够的数据进行绘图")

if __name__ == '__main__':
    main()
