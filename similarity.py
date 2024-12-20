import pandas as pd
import numpy as np
import glob
from concurrent.futures import ProcessPoolExecutor
import os
import warnings
import matplotlib.pyplot as plt
import mplfinance as mpf
from matplotlib.gridspec import GridSpec
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 5000)
warnings.filterwarnings("ignore")
from numba import jit


@jit(nopython=True)
def calculate_correlation(x, y):
    # 计算 x 和 y 的均值
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    # 计算协方差
    covariance = np.sum((x - mean_x) * (y - mean_y))
    # 计算标准差
    std_x = np.sqrt(np.sum((x - mean_x) ** 2))
    std_y = np.sqrt(np.sum((y - mean_y) ** 2))
    # 计算相关系数
    correlation = covariance / (std_x * std_y)
    return correlation


def cal_r_stock(file, stock_path, start_time, end_time, stock_name, length, df_self):
    file_name = file.split('\\')[-1].split('.')[0]
    print(f"正在处理股票: {file_name}")
    df = load_file(stock_path, file_name + '.csv')
    df = df[(df['交易日期'] >= pd.to_datetime(start_time)) & (df['交易日期'] <= pd.to_datetime(end_time))]
    if file_name == stock_name:
        for_range = len(df) - length * 2 + 1
    else:
        for_range = len(df) - length * 2 + 1 + 1
    results = []

    for i in range(for_range):
        open_corr = calculate_correlation(df['open'][i:i + length].to_numpy(), df_self['open'].to_numpy())
        close_corr = calculate_correlation(df['close'][i:i + length].to_numpy(), df_self['close'].to_numpy())
        high_corr = calculate_correlation(df['high'][i:i + length].to_numpy(), df_self['high'].to_numpy())
        low_corr = calculate_correlation(df['low'][i:i + length].to_numpy(), df_self['low'].to_numpy())
        r = (open_corr + close_corr + high_corr + low_corr) / 4
        results.append({
            'stock': file_name,
            'startdate': df['交易日期'].iloc[i],
            'enddate': df['交易日期'].iloc[i + length - 1],
            'r': r
        })
    return results


def calculate_similarity(length, start_time, end_time, stock_name, stock_start_time, stock_end_time, index_df,
                         stock_path, max_num, least_r, only_self=False):
    """
    计算指定股票与其他股票的K线相似度。

    参数:
    - length: K线的长度（天数）。
    - start_time: 检索的开始时间（格式为 'YYYYMMDD'）。
    - end_time: 检索的结束时间（格式为 'YYYYMMDD'）。
    - stock_name: 需要计算相似度的股票名（如 'sh000001'）。
    - stock_end_time: 要比较的股票数据的截止时间。
    - stock_path: 存储股票数据的路径。
    - max_num: 最多返回的相似度个数。
    - least_r: 最低相似度阈值。

    返回:
    - 返回符合相似度条件的 DataFrame。
    """
    # 判断是用length 还是 stock_end_time
    if stock_start_time is not None:
        length = len(index_df[(index_df['candle_end_time'] > pd.to_datetime(stock_start_time)) &
                              (index_df['candle_end_time'] <= pd.to_datetime(stock_end_time))])
    else:
        stock_start_time = index_df[index_df['candle_end_time'] <= pd.to_datetime(stock_end_time)].iloc[-length][
            'candle_end_time']

    # 读取基准股票数据
    base_df = load_file(stock_path, stock_name + '.csv')
    base_df = base_df[base_df['交易日期'] <= pd.to_datetime(stock_end_time)]
    df_self = base_df.tail(length)

    file_list = glob.glob(fr'{stock_path}*.csv')
    if only_self:
        file_list = [stock_name + '.csv']
    all_results = []

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # 提交每个文件到进程池
        futures = [
            executor.submit(cal_r_stock, file, stock_path, start_time, end_time, stock_name, length, df_self)
            for file in file_list
        ]
        # 收集所有结果
        for future in futures:
            all_results.extend(future.result())
    results = all_results

    # 将结果整理为 DataFrame 并按相似度排序
    df_result = pd.DataFrame(results)
    df_result = df_result.sort_values(by='r', ascending=False)

    # 删掉r太小的
    df_result = df_result[df_result['r'] >= least_r]
    # 根据指数的时间排序删掉中间有间断的
    df_result['start_idx'] = df_result['startdate'].apply(lambda x: index_df['candle_end_time'].searchsorted(x))
    df_result['end_idx'] = df_result['enddate'].apply(lambda x: index_df['candle_end_time'].searchsorted(x))
    df_result['days_diff_index'] = df_result['end_idx'] - df_result['start_idx'] + 1
    df_result = df_result[df_result['days_diff_index'] == length]
    # 留下符合条件的相似度，且最大为max_num 条的数据
    df_result = df_result.head(max_num)
    # 保留自己的结果
    df_result.loc[len(df_result)] = [stock_name, stock_start_time, stock_end_time, 100, None, None, None]
    df_result = df_result.sort_values(by='r', ascending=False)
    df_result['startdate'] = pd.to_datetime(df_result['startdate'])
    df_result['enddate'] = pd.to_datetime(df_result['enddate'])
    print(f"最终结果:\n{df_result}")
    return df_result


def load_file(path, file):
    """
    加载数据文件
    """
    path = os.path.join(path, file)
    df = pd.read_csv(path, encoding='gbk', parse_dates=['candle_end_time'])
    
    # 统一重命名列
    df.rename(columns={
        'candle_end_time': '交易日期',
        'open': '开盘价',
        'high': '最高价',
        'low': '最低价',
        'close': '收盘价',
        'amount': '成交量'  # 使用amount作为成交量
    }, inplace=True)
    
    # 计算前收盘价
    df['前收盘价'] = df['收盘价'].shift()
    df['前收盘价'].fillna(value=df['开盘价'], inplace=True)
    
    # 添加股票信息
    df['股票名称'] = file.split('.')[0]
    df['股票代码'] = file.split('.')[0]
    
    # 添加计算用的列
    df['open'] = df['开盘价']
    df['high'] = df['最高价']
    df['low'] = df['最低价']
    df['close'] = df['收盘价']
    
    return df


def merge_index(df, index_df):
    """
    将股票数据和指数合并
    """
    # ===将股票数据和指数合并，结果已经排序
    df = pd.merge(left=df, right=index_df, on='交易日期', how='right', sort=True, indicator=True)

    # ===对开、高、收、低、前收盘价价格进行补全处理
    # 用前一天的收盘价，补全收盘价的空值
    df['收盘价'].fillna(method='ffill', inplace=True)
    # 用收盘价补全开盘价、最高价、最低价的空值
    df['开盘价'].fillna(value=df['收盘价'], inplace=True)
    df['最高价'].fillna(value=df['收盘价'], inplace=True)
    df['最低价'].fillna(value=df['收盘价'], inplace=True)
    # 补全前收盘价
    df['前收盘价'].fillna(value=df['收盘价'].shift(), inplace=True)
    # 补全成交量
    df['成交量'].fillna(0, inplace=True)

    # ===用前一天的数据，补全其余空值
    df.fillna(method='ffill', inplace=True)

    # ===去除上市之前的数据
    df = df[df['股票代码'].notnull()]

    # ===判断计算当天是否交易
    df['是否交易'] = 1
    df.loc[df['_merge'] == 'right_only', '是否交易'] = 0
    del df['_merge']

    df.reset_index(drop=True, inplace=True)

    return df


def rehabilitation(df: pd.DataFrame) -> object:
    """
    计算复权价格
    """
    # =计算涨跌幅
    df['涨跌幅'] = df['收盘价'] / df['前收盘价'] - 1
    # =计算复权价：计算所有因子当中用到的价格，都使用复权价
    df['复权因子'] = (1 + df['涨跌幅']).cumprod()
    df['收盘价_复权'] = df['复权因子'] * (df.iloc[0]['收盘价'] / df.iloc[0]['复权因子'])
    df['开盘价_复权'] = df['开盘价'] / df['收盘价'] * df['收盘价_复权']
    df['最高价_复权'] = df['最高价'] / df['收盘价'] * df['收盘价_复权']
    df['最低价_复权'] = df['最低价'] / df['收盘价'] * df['收盘价_复权']
    # 原始的价格叫xx_原，复权两个字去掉
    df.rename(columns={
        '开盘价': '开盘价_原',
        '最高价': '最高价_原',
        '最低价': '最低价_原',
        '收盘价': '收盘价_原'
    }, inplace=True)
    df.rename(columns={
        '开盘价_复权': '开盘价',
        '最高价_复权': '最高价',
        '最低价_复权': '最低价',
        '收盘价_复权': '收盘价'
    }, inplace=True)

    return df


def process_r_file(f, file_path, index_df, day_list, enddate_list):
    print(f"Processing file: {f}")  # 打印正在处理的文件名
    # 加载数据
    df = load_file(file_path, f)
    # 使用指数数据与股票数据合并，补充停牌的数据
    df = merge_index(df, index_df)
    # 计算交易天数
    df['交易天数'] = df.index + 1

    if df.empty:
        return None
    df['signal'] = None
    df.loc[df['交易日期'].isin(enddate_list),'signal'] = 1
    # =====,计算你需要的技术指标
    # 计算未来表现
    for day in day_list:
        df['%s日后涨跌幅' % day] = df['收盘价'].shift(0 - day) / df['开盘价'].shift(-1) - 1
        df['%s日后_相对指数涨跌幅' % day] = df['收盘价'].shift(0 - day) / df['开盘价'].shift(-1) - df[
            '指数收盘价'].shift(0 - day) / df['指数开盘价'].shift(-1)
        df['%s日后是否上涨' % day] = df['%s日后涨跌幅' % day] > 0
        df['%s日后是否上涨' % day].fillna(value=False, inplace=True)

    # 选取指定时间范围内的股票
    # 删除一些有信号但实际无法买入的
    df['event'] = 1
    df = df[df['成交量'] > 0]
    df = df[df['signal'].notna()]
    new_columns = ['signal', '交易日期', '股票代码', 'event']
    for day in day_list:
        new_columns.extend([
            '%s日后涨跌幅' % day,
            '%s日后_相对指数涨跌幅' % day,
            '%s日后是否上涨' % day
        ])

    df = df[new_columns]
    return df


def analysis_for_r(all_df, day_list):
    results = []
    # 计算N日后涨跌幅大于0的概率
    result = {}
    result['信号1出现次数'] = all_df['signal'].sum()
    result['信号0出现次数'] = (1 - all_df['signal']).sum()
    for signal, group in all_df.groupby('signal'):
        signal_type = '看涨信号' if signal == 1 else '看跌信号'
        print('\n', '=' * 10, signal_type, '=' * 10)
        result['signal'] = signal_type  # 先存储信号类型
        for i in day_list:
            if signal == 1:
                prob = float(group[group[str(i) + '日后涨跌幅'] > 0].shape[0]) / group.shape[0]
                up_return = float(group[group[str(i) + '日后涨跌幅'] > 0][str(i) + '日后涨跌幅'].mean())
                down_return = float(group[group[str(i) + '日后涨跌幅'] < 0][str(i) + '日后涨跌幅'].mean())
                avg_return = float(group[group[str(i) + '日后涨跌幅'] > 0].shape[0] / group.shape[0] *
                                   group[group[str(i) + '日后涨跌幅'] > 0][str(i) + '日后涨跌幅'].mean() +
                                   group[group[str(i) + '日后涨跌幅'] <= 0][str(i) + '日后涨跌幅'].mean() *
                                   group[group[str(i) + '日后涨跌幅'] <= 0].shape[0] / group.shape[0])
                ex_index_return = float(group[str(i) + '日后_相对指数涨跌幅'].mean())
                print(
                    f'{i}天后涨跌幅大于0概率\t {prob}\t {i}天后上涨收益率\t {up_return}\t {i}天后下跌收益率\t {down_return}\t {i}天后每笔交易平均收益率\t {avg_return}')

            else:
                prob = float(group[group[str(i) + '日后涨跌幅'] < 0].shape[0]) / group.shape[0]
                up_return = float(group[group[str(i) + '日后涨跌幅'] > 0][str(i) + '日后涨跌幅'].mean())
                down_return = float(group[group[str(i) + '日后涨跌幅'] < 0][str(i) + '日后涨跌幅'].mean())
                avg_return = float(group[group[str(i) + '日后涨跌幅'] > 0].shape[0] / group.shape[0] *
                                   group[group[str(i) + '日后涨跌幅'] > 0][str(i) + '日后涨跌幅'].mean() +
                                   group[group[str(i) + '日后涨跌幅'] <= 0][str(i) + '日后涨跌幅'].mean() *
                                   group[group[str(i) + '日后涨跌幅'] <= 0].shape[0] / group.shape[0])
                ex_index_return = float(group[str(i) + '日后_相对指数涨跌幅'].mean())
                print(
                    f'{i}天后涨跌幅小于0概率\t {prob}\t {i}天后上涨收益率\t {up_return}\t {i}天后下跌收益率\t {down_return}\t {i}天后每笔交易平均收益率\t {avg_return}')

            # 为每个i天的结果生成相应的列名
            result[f'{i}天后大于0概率'] = prob
            result[f'{i}天后上涨收益率'] = up_return
            result[f'{i}天后下跌收益率'] = down_return
            result[f'{i}天后每笔交易平均收益率'] = avg_return
            result[f'{i}日后_相对指数涨跌幅'] = ex_index_return

        # 将结果存储到列表中
        results.append(result)

    result_df = pd.DataFrame(results)
    return result_df


def cal_for_pic(row, file_path, future_days):
    f = row['stock'] + '.csv'
    print(f"Processing file: {f}")  # 打印正在处理的文件名

    df = load_file(file_path, f)
    df_list = []
    for i in range(len(row['enddate'])):
        df_tep = df.copy()
        enddate = row['enddate'][i]
        startdate = row['startdate'][i]
        r = row['r'][i]
        df_tep = df_tep[(df_tep['交易日期'].shift(future_days) <= pd.to_datetime(enddate)) &
                (df_tep['交易日期'] >= pd.to_datetime(startdate))]
        df_tep['enddate'] = enddate
        df_tep['startdate'] = startdate
        df_tep['r'] = r
        df_list.append(df_tep)
    df = pd.concat(df_list,axis=0)
    # ===使用指数数据与股票数据合并，补充停牌的数据
    df = df[['股票代码', '股票名称', '交易日期', 'open', 'high', 'low', 'close', 'enddate', 'startdate', 'r']]
    # 掩盖掉 参考股票的未来数据
    filtered_df = df[df['r'] == 100]
    last_k_indices = filtered_df.tail(future_days).index
    df.loc[last_k_indices, 'open'] = None
    df.loc[last_k_indices, 'high'] = None
    df.loc[last_k_indices, 'close'] = None
    df.loc[last_k_indices, 'low'] = None

    return df


def draw_data(data, future_pics, length, stock_start_time, stock_end_time, future_days, index_df):
    future_pics = future_pics + 1
    if stock_start_time is not None:
        length = len(index_df[(index_df['交易日期'] >= pd.to_datetime(stock_start_time)) &
                              (index_df['交易日期'] <= pd.to_datetime(stock_end_time))])

    fixed_k_lines = length + future_days
    # 确保交易日期为 datetime 格式
    data['交易日期'] = pd.to_datetime(data['交易日期'])
    data['startdate'] = pd.to_datetime(data['startdate'])

    # 获取唯一的股票代码和开始日期组合
    unique_groups = data.groupby(['股票代码', 'startdate'], sort=False).size().index[:future_pics]
    n_stocks = len(unique_groups)
    # n_stocks = future_pics
    print(10 * '=', f'最相似的{n_stocks - 1}', 10 * '=')

    # 设置绘图大小及网格布局
    fig = plt.figure(figsize=(10, 3 * n_stocks))
    gs = GridSpec(n_stocks, 1)

    # 自定义mplfinance样式
    mc = mpf.make_marketcolors(up='red', down='green', inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc)

    # 绘制每个股票的K线图，从 startdate 开始展示其后的表现
    for i, (stock_code, startdate) in enumerate(unique_groups):
        print(stock_code, startdate)

        # 获取每只股票从 startdate 开始的交易数据
        stock_data = data[(data['股票代码'] == stock_code) & (data['startdate'] == startdate)].copy()

        # 检查当前数据的行数
        current_rows = len(stock_data)

        # 如果数据的行数少于固定的K线数量，则填充额外的日期和空值
        if current_rows < fixed_k_lines:
            missing_rows = fixed_k_lines - current_rows
            last_date = stock_data['交易日期'].max()
            new_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=missing_rows, freq='D')

            # 创建空值的 DataFrame，只有交易日期有值，其他列为空
            empty_data = pd.DataFrame({
                'open': np.nan,
                'high': np.nan,
                'low': np.nan,
                'close': np.nan,
                '交易日期': new_dates
            })

            # 将补充的空数据和原始数据合并
            stock_data = pd.concat([stock_data, empty_data], ignore_index=True)

        # 准备数据以供 mplfinance 绘图
        stock_data.set_index('交易日期', inplace=True)
        stock_data = stock_data[['open', 'high', 'low', 'close', 'enddate', 'startdate', 'r']]

        ax = fig.add_subplot(gs[i, 0])

        # 使用自定义样式绘制K线图
        mpf.plot(stock_data, type='candle', ax=ax, style=s)
        # 设置标题和标签
        enddate = stock_data['enddate'].iloc[0]
        R = round(stock_data['r'].iloc[0], 4)
        enddate = pd.to_datetime(enddate).date()
        if R == 100:
            R = 'origin'
        ax.set_title(f'{stock_code} (Start: {startdate.date()}  End: {enddate}) R = {R}', loc='center')  # 标题位置居中

        ax.set_ylabel('Price')
        # 隐藏x轴标签
        ax.set_xticklabels([])

        # 在第10根k线与第11根k线之间画一条竖线
        ax.axvline(x=length - 0.5, color='blue', linestyle='--', linewidth=1)

    # 调整子图之间的间距
    plt.subplots_adjust(hspace=1)  # hspace 设置垂直间距

    # 保存图表到文件
    output_filename = f'stock_analysis_plots_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(output_filename)
    plt.close()
    print(f"图表已保存为 {output_filename}")

