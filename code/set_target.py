# Pandas
import numpy as np
import pandas as pd
import datetime
from numba import njit, prange

def detect_trend(df, cost=0.00072):
    trend_returns = [None] * len(df)  # 预初始化输出序列
    i = 0
    n = len(df)
    lowest_index = 0
    highest_index = 0
    while i < n:
        '''if i == 0:
            print('i:',i)'''
        open_price = df.iloc[i]['open']
        for close_index in range(i, n):
            '''if i == 0:
                print('close_index:', close_index)'''
            close_price = df.iloc[close_index]['close']
            price_change = abs(close_price / open_price - 1)
            # 当波动不足时继续寻找
            if price_change <= cost:
                close_index += 1
                continue
            # 当波动超过cost时确认趋势
            else:
                if close_price > open_price:  # 0: 上涨, 1: 下跌
                    highest_index = close_index
                    highest_price = close_price
                    # 2. 继续寻找趋势结束点
                    for extreme in range(close_index, n):
                        current_close = df.iloc[extreme]['close']
                        '''if i == 0:
                            print('上涨extreme:',extreme)
                            print(current_close, highest_price)'''
                        if current_close > highest_price:
                            highest_index = extreme
                            highest_price = current_close
                        # 如果回撤超过交易成本代表趋势结束，填上最大值到之前所有位置的趋势数
                        if (highest_price - current_close) / highest_price > cost:
                            '''if i == 0:
                                print(i, highest_index)'''
                            for trend_index in range(i, highest_index + 1):
                                '''trend_returns[trend_index] = ((highest_price - df.iloc[trend_index]['open']) /
                                                              df.iloc[trend_index]['open'])'''
                                trend_returns[trend_index] = -1  # 将输出修改为分类用的0, 1格式
                                '''if i == 0:
                                    print('trend_returns[0]:', trend_returns[trend_index])
                            if i == 0:
                                print('gggg')'''
                            break  # 出现 0.036% 的回撤，结束趋势
                    break
                else:
                    lowest_index = close_index
                    lowest_price = close_price
                    # 2. 继续寻找趋势结束点
                    for extreme in range(close_index, n):
                        '''if i == 0:
                            print('下跌extreme:', extreme)'''
                        current_close = df.iloc[extreme]['close']
                        if current_close < lowest_price:
                            lowest_index = extreme
                            lowest_price = current_close
                        # 如果回撤超过交易成本代表趋势结束，填上最小值到之前所有位置的趋势数
                        if (current_close - lowest_price) / lowest_price > cost:
                            for trend_index in range(i, lowest_index + 1):
                                '''trend_returns[trend_index] = ((lowest_price - df.iloc[trend_index]['open']) /
                                                              df.iloc[trend_index]['open'])'''
                                trend_returns[trend_index] = 1  # 将输出修改为分类用的0, 1格式
                            break
                    break
        i = max(lowest_index, highest_index) + 1  # 移动 i 到下一个未处理的位置

    # 将趋势数据添加到 df
    df['trend_returns'] = trend_returns

    # 获取当前时间
    '''current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # 生成文件名并保存 CSV
    file_name = f"trend_data_{current_time}.csv"
    df.to_csv(file_name, index=False, encoding='utf-8')

    print(f"文件已保存为: {file_name}")'''

    return df


@njit(parallel=True)  # 使用 JIT 编译 + 并行计算
def detect_trend_numba(open_prices, close_prices, cost):
    """
    使用 numba 加速的趋势检测函数
    """
    n = len(open_prices)
    trend_returns = np.full(n, np.nan)  # 预初始化 NaN 数组
    i = 0
    highest_index = 0
    lowest_index = 0
    while i < n:
        open_price = open_prices[i]
        for close_index in range(i, n):
            close_price = close_prices[close_index]
            price_change = abs(close_price / open_price - 1)

            if price_change <= cost:
                continue  # 波动不足，继续寻找

            # 确定趋势方向
            if close_price > open_price:
                # 上涨趋势
                highest_index = close_index
                highest_price = close_price

                for extreme in range(close_index, n):
                    current_close = close_prices[extreme]

                    if current_close > highest_price:
                        highest_index = extreme
                        highest_price = current_close

                    if (highest_price - current_close) / highest_price > cost:
                        trend_returns[i:highest_index + 1] = 1  # 记录趋势
                        break
                break

            else:
                # 下跌趋势
                lowest_index = close_index
                lowest_price = close_price

                for extreme in range(close_index, n):
                    current_close = close_prices[extreme]

                    if current_close < lowest_price:
                        lowest_index = extreme
                        lowest_price = current_close

                    if (current_close - lowest_price) / lowest_price > cost:
                        trend_returns[i:lowest_index + 1] = 0  # 记录趋势
                        break
                break
        if i == max(highest_index, lowest_index) + 1:
            break
        else:
            i = max(highest_index, lowest_index) + 1  # 移动索引

    return trend_returns


def detect_trend_optimized(df, cost=0.05):  # cost=0.00072
    open_prices = df['open'].values
    close_prices = df['close'].values

    print("开始调用优化的 detect_trend_numba 函数...")
    # 计算趋势
    trend_returns = detect_trend_numba(open_prices, close_prices, cost)
    print("趋势数据开始添加到 DataFrame")
    # 存入 DataFrame
    df['trend_returns'] = trend_returns

    return df
