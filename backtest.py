import pandas as pd
import numpy as np
import math
from math import fabs

from datetime import date, datetime

import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

# 数据基类
class Data():
    def __init__(self, path):
        # 导入基础数据
        self.close = pd.read_csv(path + "qfq/close_of_all_stocks.csv",
                                 index_col=0)
        self.pct_chg = pd.read_csv(path + "qfq/pctChg_of_all_stocks.csv",
                                  index_col=0)
        self.tradestatus = pd.read_csv(path + "qfq/tradestatus_of_all_stocks.csv",
                                  index_col=0)
        self.isST = pd.read_csv(path + "qfq/isST_of_all_stocks.csv",
                                  index_col=0)



# 策略基类
class Strategy():

    # 用构造因子需要的数据初始化
    def __init__(self, data):
        self.data = data

    # 一些工具函数

    #
    def rank(self, frame):
        return frame.rank(axis=1, method='first', pct=True)
        # return frame.rank(axis=1, method='first')
        # return frame.rank(axis=1,ascending=False,method='first')

    # 将数据下移一个日期
    def delay(self, frame, days):
        return frame.shift(days)

    # 求两个dataframe的相关系数
    def correlation(self, frame1, frame2, days):
        # dataframe的apply函数就是对所有元素进行同一种函数操作的意思
        # frame1.apply(lambda x:x.astype(float))
        # frame2.apply(lambda x:x.astype(float))
        # print(frame1.info())
        # print(frame2.info())
        return frame1.rolling(days).corr(frame2)

    # 求两个dataframe的协方差
    def covariance(self, frame1, frame2, days):
        return frame1.rolling(days).cov(frame2)

    # 这个函数是针对整体的,这里的a不表示天数，而是表示整体放缩至的大小
    def scale(self, frame, a=1):
        return frame.mul(a).div(np.abs(frame).sum())

    # 计算dataframe在时间上的差值
    def delta(self, frame, days):
        return frame.diff(days)

    def decay_linear(self, frame, days):
        return frame.rolling(days).apply(self.decay_linear_apply)

    def decay_linear_apply(self, slice):
        # print(slice)
        # 加1是为了之后创建list
        days = len(slice) + 1
        # print(days)
        weight = np.arange(1, days, 1)
        weight = weight / weight.sum()
        return (slice * weight).sum()

    # 计算在时间上的最小值
    def ts_min(self, frame, days):
        return frame.rolling(days).min()

    # 计算在时间上的最大值
    def ts_max(frame, days):
        return frame.rolling(days).max()

    # 计算在时间上的排名（从小到大）
    def ts_argmin(self, frame, days):
        return frame.rolling(days).apply(np.argmin)

    # 计算在时间上的排名（从大到小）
    def ts_argmax(self, frame, days):
        return frame.rolling(days).apply(np.argmax)

    def ts_rank(self, frame, days):
        return frame.rolling(days).apply(self.ts_rank_apply)

    def ts_rank_apply(self, slice):
        return list(slice.argsort()).index(4)

    def sum(self, frame, days):
        return frame.rolling(days).sum()

    # 累乘
    def product(self, frame, days):
        return frame.rolling(days).apply(np.prod)

    # 标准差
    def stddev(self, frame, days):
        return frame.rolling(days).std()

    # 几日均值
    def sma(self, frame, days):
        return frame.rolling(days).mean()


    # 这里可以定义一个专门的实现具体策略的函数或者factor，然后创建新策略的时候就直接继承Stragey然后
    # 实现具体的因子函数就好
    def factor(self):
        pass


class simple_Strategy(Strategy):
    def factor(self):
        # pct_Chg = self.data.pct_chg
        # close = self.data.close
        volume = self.data.volume
        return self.rank(volume)


class Backtest():
    def __init__(self, buy_price, buy_price_pctChg, capital, tran_cost):
        # 以比如收盘价或者开盘价等作为买入价格
        self.buy_price = buy_price
        # 买入价格的价格变化
        self.buy_price_pctChg = buy_price_pctChg.fillna(0)
        self.up_limit = None
        self.down_limit = None

        # 初始总资金
        self.capital = capital
        # 当前持有股票的series
        self.stocks_hold = pd.Series()
        # 现金剩余
        self.remain = 0
        # 当前总价值
        self.total_value = 0
        # 历史总价值列表
        self.his_tot_val = []

        # 交易手续费
        self.tran_cost = tran_cost

        # 进行画图与指标测算时需要用到的数据
        path = "F:/Python/backtest_/"
        self.hs300 = pd.read_excel(path + "index/沪深300指数.xlsx", index_col="Date")


    # signal是策略类输出的信号矩阵，start_date是起始时间，end_date是结束时间，stock_num指选择的股票数目，period是调仓周期
    def caculate_result(self, signal, start_date="", end_date="", stock_num=30, period=""):
        # 当天只能用前一天的信号
        signal = signal.shift(1)
        # 按初始结束日期截取信号矩阵
        signal = signal.loc[start_date:end_date]
        up_limit = (self.buy_price.shift(1) * 1.1).loc[start_date:end_date]
        down_limit = (self.buy_price.shift(1) * 0.9).loc[start_date:end_date]
        self.buy_price = self.buy_price.loc[start_date:end_date]

        # 开始回测 一开始所有资金都没有买股票，所以现金剩余为全部资产
        self.remain = self.capital
        count = 0
        for date, target in signal.iterrows():
            buy_price_daily = self.buy_price.loc[date]
            up_limit_daily = up_limit.loc[date]
            down_limit_daily = down_limit.loc[date]

            # 首先计算上一期持仓的股票在这一期总价值变成了多少
            buy_price_pctChg_daily = self.buy_price_pctChg.loc[date]
            self.stocks_hold = (self.stocks_hold * (buy_price_pctChg_daily[self.stocks_hold.index] * 0.01 + 1))
            self.total_value = self.stocks_hold.sum() + self.remain
            self.his_tot_val.append(self.total_value/self.capital)
            print("日期:" + str(date))
            print("组合总价值：" + str(self.total_value))
            print("组合持有情况：")
            print(self.stocks_hold)
            print("现金情况："  + str(self.remain))
            # 如果是最后一个日期，不需要换仓
            if date == end_date:
                break
            # 如果不是换仓日期，则不换仓
            if count % period == 0:
                count += 1
            else:
                count += 1
                continue

            # 接着进行调仓
            target = target.nlargest(stock_num, keep="first")
            # 未分配的股票数目
            stock_unallocated_num = stock_num
            # 理论上每只股票能分到的持有量
            target_value = self.total_value * (1 - 2 * self.tran_cost)/stock_unallocated_num

            # 先卖出不在新的目标股票中的股票
            diff = self.stocks_hold.loc[self.stocks_hold.index.difference(target.index)].sort_values(ascending=False)
            print("卖出部分：" )
            print(diff)
            for stock, position in diff.items():
                if buy_price_daily[stock] > down_limit_daily[stock]:
                    self.stocks_hold.drop(stock, inplace=True)
                    # 现金加上卖出得到的钱，同时减去手续费
                    self.remain += position * (1 - self.tran_cost)
                else:
                    print(str(stock) + "由于跌停，无法卖出")
                    # 跌停所以剩下的股票不能分那么多资金，每只股票平摊由于卖不出去占用的资金
                    target_value -= position / stock_unallocated_num

            # 再调整既在原有持仓又在目标中的股票的持有量
            inter = self.stocks_hold.loc[self.stocks_hold.index.intersection(target.index)].sort_values(ascending=False)
            print("调整部分：" )
            print(inter)
            for stock, position in inter.items():
                delta = target_value - position
                if delta < 0:
                    if buy_price_daily[stock] > down_limit_daily[stock]:
                        # 现金加上卖出得到的钱，同时减去手续费
                        self.remain += (-1*delta) * (1 - self.tran_cost)
                        # 股票持有量变化
                        self.stocks_hold[stock] += delta
                        # 未分配股票数量减去1
                        stock_unallocated_num -= 1
                    else:
                        print(str(stock) + "由于跌停，无法卖出")
                        # 跌停无法卖出导致理论可用资金量减少，生成一个新的target_value
                        stock_unallocated_num -= 1
                        # 跌停所以剩下的股票不能分那么多资金，每只股票平摊由于卖不出去占用的资金
                        target_value -= (-1 * delta)/stock_unallocated_num
                else:
                    if buy_price_daily[stock] < up_limit_daily[stock]:
                        # 现金减去买入的钱，同时减去手续费（这里减去手续费不用担心不够钱，因为之前设置target_value时已经考虑到了）
                        # rem是小于100股的无法买入的部分
                        rem = target_value % (buy_price_daily[stock] * 100)
                        delta -= rem
                        self.remain -= (delta * (1 + self.tran_cost) - rem)
                        # 股票持有量变化
                        self.stocks_hold[stock] += delta
                        # 未分配股票数量减去1
                        stock_unallocated_num -= 1
                    else:
                        print(str(stock) + "由于涨停，无法买入")
                        stock_unallocated_num -= 1
                        # 涨停所以这只股票占用不了那么多资金，省下每只股票可以多分一点
                        target_value += delta/stock_unallocated_num

            # 最后买入stocks_hold中没有的股票
            new = target.loc[target.index.difference(self.stocks_hold.index)].sort_values(ascending=False)
            print("买入部分：" )
            print(new)
            for stock, ratio in new.items():
                if buy_price_daily[stock] < up_limit_daily[stock]:
                    delta = target_value
                    # rem是小于100股的无法买入的部分
                    rem = target_value % (buy_price_daily[stock] * 100)
                    delta -= rem
                    # 现金减去买入的钱，同时减去手续费（这里减去手续费不用担心不够钱，因为之前设置target_value时已经考虑到了）
                    self.remain -= (delta * (1 + self.tran_cost) - rem)
                    # 股票持有量变化
                    self.stocks_hold[stock] = delta
                    stock_unallocated_num -= 1
                else:
                    print(str(stock) + "由于涨停，无法买入")
                    stock_unallocated_num -= 1
                    # 涨停所以这只股票占用不了那么多资金，省下每只股票可以多分一点
                    target_value += target_value / stock_unallocated_num
            print("调仓后组合情况：")
            print(self.stocks_hold)

        # 指标测算与画图
        self.caculate_indicator(start_date, end_date)
        self.return_rate_chart(start_date, end_date)

    # 计算最大回撤
    def MaxDrawdown(self, df_tot_val):
        '''最大回撤率'''
        i = (((np.maximum.accumulate(df_tot_val) - df_tot_val) / np.maximum.accumulate(df_tot_val)).idxmax())[0]  # 结束位置
        if i == 0:
            return 0
        j = ((df_tot_val.iloc[:i, :]).idxmax())[0]  # 开始位置
        return ((df_tot_val.iloc[j, :] - df_tot_val.iloc[i, :]) / (df_tot_val.iloc[j, :]))[0]

    def caculate_indicator(self, start_date, end_date):
        result = pd.Series()
        print(self.his_tot_val)
        print("总收益：" + str(self.his_tot_val[-1] - 1))
        a = datetime.strptime(start_date, '%Y%m%d')
        b = datetime.strptime(end_date, '%Y%m%d')
        annu_return = math.pow((self.his_tot_val[-1]), 365.25 / (b - a).days) - 1
        print("年化收益率：", annu_return)
        result.loc["年化收益率"] = annu_return
        df_tot_val = pd.DataFrame(self.his_tot_val)
        df_return = ((df_tot_val/df_tot_val.shift(axis=0)-1).drop(0))
        annu_vol = (df_return.std()[0]) * math.sqrt(250)
        sharp = (annu_return - 0.04) / annu_vol
        result.loc["夏普比率"] = sharp
        print("夏普比率：", sharp)
        result.loc["年化波动率"] = annu_vol
        print("年化波动率：", annu_vol)
        maxDrawdown = self.MaxDrawdown(df_tot_val)
        result.loc["最大回撤"] = maxDrawdown
        print("最大回撤：", maxDrawdown)
        # df_hs300 = pd.DataFrame(self.hs300)
        # df_hs300 = self.hs300
        # print(df_hs300)
        # df_hs300_return = ((df_hs300 / df_hs300.shift(1) - 1).drop(0))
        # rel_vol = ((df_return - df_hs300_return).std()[0]) * math.sqrt(250)
        # hs300_annu_return = math.pow((self.hs300[-1]), 365.25 / (a - b).days) - 1
        # infor_ratio = (annu_return - hs300_annu_return) / rel_vol
        # result.loc["信息比率"] = infor_ratio
        # print("信息比率：", infor_ratio)
        # ic = infor_ratio / math.sqrt((a - b).days)
        # # print("信息系数：",ic)
        # beta = (pd.concat([df_return, df_hs300_return], axis=1).cov()).iloc[0, 1] / df_hs300_return.var()[0]
        # result.loc["Beta"] = beta
        # print("Beta：", beta)
        # alpha = (annu_return - 0.04) - beta * (hs300_annu_return - 0.04)
        # result.loc["alpha"] = alpha
        # print("alpha:", alpha)

    def return_rate_chart(self, start_date, end_date):
        hs300 = self.hs300.loc[start_date, end_date]
        dates = hs300.index
        xs = [datetime.strptime(str(d), '%Y%m%d').date() for d in dates]
        print(xs)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y%m%d'))
        time = range(0, len(self.his_tot_val))
        plt.figure(figsize=(20, 10))

        def to_percent(temp, position):
            return '%1.0f' % (100 * temp) + '%'

        plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
        plt.xlabel("dates")
        plt.ylabel("ROR")
        plt.plot(xs, self.his_tot_val, label="value-driven growth-focused investment strategy")
        plt.plot(xs, hs300, label="CSI300")
        plt.legend()

