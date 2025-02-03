# Pandas
import pandas as pd

import datetime

# load the data
data = pd.read_csv("../input/btcusdt/okex_btcusdt_kline_1m.csv")

# Finta needs specific columns' names to work
data_finta = pd.DataFrame()
data_finta['open'] = data['o']
data_finta['high'] = data['h']
data_finta['low'] = data['l']
data_finta['close'] = data['c']
data_finta['volume'] = data['v']


def detect_trend(df, cost=0.00036):
    trend_returns = [None] * len(df)  # 预初始化输出序列
    i = 0
    n = len(df)
    lowest_index = 0
    highest_index = 0
    while i < n:
        open_price = df.iloc[i]['open']
        for close_index in range(i, n):
            close_price = df.iloc[close_index]['close']
            price_change = abs(close_price / open_price - 1)
            # 当波动不足时继续寻找
            if price_change <= cost:
                close_index += 1
                continue
            # 当波动超过cost时确认趋势
            else:
                if close_price > open_price:  # 1: 上涨, -1: 下跌
                    highest_index = close_index
                    highest_price = close_price
                    # 2. 继续寻找趋势结束点
                    for extreme in range(close_index, n):
                        current_close = df.iloc[extreme]['close']
                        if current_close > highest_price:
                            highest_index = extreme
                            highest_price = current_close
                        # 如果回撤超过交易成本代表趋势结束，填上最大值到之前所有位置的趋势数
                        if (highest_price - current_close) / highest_price > cost:
                            for trend_index in [i, highest_index]:
                                trend_returns[trend_index] = (highest_price - df.iloc[trend_index]['close']) / df.iloc[trend_index]['close']
                            break  # 出现 0.036% 的回撤，结束趋势
                else:
                    lowest_index = close_index
                    lowest_price = close_price
                    # 2. 继续寻找趋势结束点
                    for extreme in range(close_index, n):
                        current_close = df.iloc[extreme]['close']
                        if current_close < lowest_price:
                            lowest_index = extreme
                            lowest_price = current_close
                        # 如果回撤超过交易成本代表趋势结束，填上最小值到之前所有位置的趋势数
                        if (current_close - lowest_price) / lowest_price > cost:
                            for trend_index in [i, lowest_index]:
                                trend_returns[trend_index] = (df.iloc[trend_index]['close'] - lowest_price) / df.iloc[trend_index]['close']
                            break
        i = max(lowest_index, highest_index) + 1  # 移动 i 到下一个未处理的位置

    # 将趋势数据添加到 df
    df['trend_returns'] = trend_returns

    # 获取当前时间
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # 生成文件名并保存 CSV
    file_name = f"trend_data_{current_time}.csv"
    df.to_csv(file_name, index=False, encoding='utf-8')

    print(f"文件已保存为: {file_name}")

    return trend_returns


df = pd.DataFrame(data_finta)

# 运行趋势检测
trend_df = detect_trend(df)
