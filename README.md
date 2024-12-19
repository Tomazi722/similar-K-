# 股票K线相似度分析

## 概述

本Python脚本实现了以下功能：

1. **数据获取**：使用`akshare`库获取历史股票或指数数据。
2. **数据处理**：处理和标准化获取的数据。
3. **相似度计算**：计算K线（蜡烛图）模式的相似度。
4. **分析**：分析相似度结果，识别模式和潜在的交易机会。
5. **可视化**：生成图表，展示相似度分析及基于历史模式的未来表现。

## 目录

- [功能](#功能)
- [安装要求](#安装要求)
- [安装步骤](#安装步骤)
- [使用方法](#使用方法)
- [功能详细介绍](#功能详细介绍)
  - [1. 数据获取](#1-数据获取)
  - [2. 相似度计算](#2-相似度计算)
  - [3. 分析](#3-分析)
  - [4. 可视化](#4-可视化)
- [配置](#配置)
- [输出结果](#输出结果)
- [常见问题](#常见问题)
- [许可证](#许可证)

## 功能

- **全面的数据获取**：获取个股和指数的历史数据。
- **K线相似度分析**：识别指定时间范围内相似的蜡烛图模式。
- **可定制的参数**：允许用户定义时间范围、股票代码、相似度阈值等。
- **自动化分析与报告**：生成CSV报告和基于分析的可视化图表。
- **高效的处理**：利用多进程（如果在`similarity`模块中实现）加速相似度计算。

## 安装要求

确保您的系统中安装了Python 3.7或更高版本。

## 安装步骤

1. **克隆仓库**

   ```bash
   git clone https://github.com/yourusername/stock-kline-similarity.git
   cd stock-kline-similarity
   ```

2. **创建虚拟环境（可选，但推荐）**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows用户: venv\Scripts\activate
   ```

3. **安装所需依赖**

   ```bash
   pip install -r requirements.txt
   ```

   *如果未提供`requirements.txt`，请手动安装必要的包：*

   ```bash
   pip install akshare pandas
   ```

   *注意：确保`similarity`模块在您的项目目录中或已正确安装。*

## 使用方法

使用Python运行脚本：

```bash
python your_script_name.py
```

*将`your_script_name.py`替换为实际的Python脚本名称。*

## 功能详细介绍

### 1. 数据获取

- **函数**：`get_stock_data(stock_code, start_date, end_date)`
  
- **描述**：根据提供的股票代码和日期范围，从AKShare获取历史股票或指数数据。

- **参数**：
  - `stock_code`（字符串）：股票代码（例如，`'sh000001'`表示上海指数）。
  - `start_date`（字符串）：开始日期，格式为`YYYY/MM/DD`。
  - `end_date`（字符串）：结束日期，格式为`YYYY/MM/DD`。

- **返回**：包含标准化列的`pandas.DataFrame`：
  - `candle_end_time`：蜡烛图结束时间戳。
  - `open`：开盘价。
  - `high`：最高价。
  - `low`：最低价。
  - `close`：收盘价。
  - `volume`：成交量。
  - `amount`：成交额。

### 2. 相似度计算

- **函数**：`calculate_similarity_process(args)`
  
- **描述**：用于调用`similarity`模块中的`calculate_similarity`函数的包装函数。

- **参数**：`calculate_similarity`函数所需的参数元组。

- **返回**：相似度计算结果的`DataFrame`。

### 3. 分析

- **过程**：
  - **步骤1**：计算K线模式之间的相似度得分。
  - **步骤2**：通过选择相似度最高的模式，准备可视化的数据。
  - **步骤3**：分析相似模式在指定未来天数内的表现。

- **关键参数**：
  - `length`：每个K线模式考虑的交易日数量。
  - `max_num`：评估的相似模式的最大数量。
  - `least_r`：最低相似度阈值（例如，0.9）。
  - `day_list`：未来天数列表，用于分析表现（例如，`[1, 2, 3, 5, 10, 20]`）。

### 4. 可视化

- **函数**：`draw_data(pic_df, future_pics, length, stock_start_time, stock_end_time, future_days, index_df)`
  
- **描述**：生成图表，展示相似K线模式在未来天数内的表现。

- **参数**：
  - `pic_df`：用于绘图的数据`DataFrame`。
  - `future_pics`：要可视化的相似模式数量。
  - `future_days`：图表中显示的未来天数。
  - 其他参数与分析配置相关。

## 配置

在`main()`函数内自定义以下参数以满足您的分析需求：

- **日期范围**：

  ```python
  start_time = '1990/01/01'
  end_time = '2024/12/18'
  ```

- **股票选择**：

  ```python
  stock_name = 'SH601118'
  ```

- **相似度计算参数**：

  ```python
  length = 20
  stock_end_time = '2024/10/17'
  stock_start_time = None  # 设置为具体日期或None
  max_num = 10000
  only_self = True  # 设置为False以在所有股票中比较
  least_r = 0.9
  day_list = [1, 2, 3, 5, 10, 20]
  future_days = 5
  future_pics = 5
  ```

- **文件路径**：
  - 脚本将在与脚本文件相同的目录中保存数据和结果。如果需要更改目录，请修改`base_path`。

## 输出结果

- **数据文件**：
  - `SH601118.csv`：包含指定股票的历史数据。
  - `D_0_相识度计算结果.csv`：存储相似度计算结果。

- **分析结果**：
  - 控制台打印，显示基于相似度得分的表现分析。

- **可视化图表**：
  - 生成展示相似K线模式未来表现的图表。具体格式和位置取决于`draw_data`函数的实现。

## 常见问题

- **数据获取问题**：
  - 确保股票代码正确且被AKShare支持。
  - 检查互联网连接，因为脚本需要在线获取数据。

- **模块未找到错误**：
  - 确认已安装所有必要的模块（`akshare`、`pandas`、`similarity`等）。
  - 确保`similarity`模块位于正确的目录中或已通过`pip`安装。

- **编码问题**：
  - 脚本以`gbk`编码保存CSV文件。如有需要，可修改`to_csv`中的`encoding`参数。

- **绘图数据不足**：
  - 确保相似度计算返回足够的数据点（`future_pics`）以进行可视化。

## 许可证

本项目采用 [MIT 许可证](LICENSE)。

---

*如有任何问题或功能请求，请在 [GitHub 仓库](https://github.com/yourusername/stock-kline-similarity/issues) 中提交问题。*
