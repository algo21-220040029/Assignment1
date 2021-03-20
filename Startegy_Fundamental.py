from backtest import *

class Fundamental_Data(Data):
    def __init__(self, path):
        Data.__init__(self, path)
        #     资产负债率
        self.DEBT = pd.read_csv(path + "财务数据\资产负债率_已处理.csv", index_col=0)
        #     Operating Cash Flow Increasing
        self.OCF_incr = pd.read_csv(path + "财务数据\经营现金流量净额增长_已处理.csv",
                                    index_col=0)
        #     扣非净资产收益率
        self.WROE = pd.read_csv(path + "财务数据\扣非净资产收益率_已处理.csv", index_col=0)
        #     营业收入增长
        self.OPE_REV_INC = pd.read_csv(path + "财务数据\营收增长_已处理.csv",
                                       index_col=0)
        #     销售毛利率
        self.GPS = pd.read_csv(path + "财务数据\销售毛利率_已处理.csv", index_col=0)

data = Fundamental_Data(path="F:/Python/backtest_/")

class fundamantal_Strategy(Strategy):

    def factor(self):
        tradestatus = ((self.data.tradestatus).fillna(0))
        isST = ((self.data.isST).fillna(0))
        DEBT = self.data.DEBT.sort_index(axis=1)
        OCF_incr = self.data.OCF_incr.sort_index(axis=1)
        WROE = self.data.WROE.sort_index(axis=1)
        OPE_REV_INC = self.data.OPE_REV_INC.sort_index(axis=1)
        GPS = self.data.GPS.sort_index(axis=1)

        #         return (self.rank(DEBT).multiply(-0.2)).add(tradestatus.multiply(50000))
        #         return self.rank(OPE_REV_INC).multiply(5).add(self.rank(DEBT).multiply(-0.2)).add(self.rank(WROE).multiply(0.2)).add(self.rank(OCF_incr).multiply(3.4)).add(tradestatus.multiply(50000))
        return self.rank(OPE_REV_INC).multiply(5).add(self.rank(DEBT.multiply(-0.2))). \
            add(self.rank(WROE.multiply(0.2))).add(self.rank(OCF_incr.multiply(3.4))).\
            multiply(tradestatus).multiply((isST-1)*(-1)). \
            multiply((OPE_REV_INC <= 350)).multiply((OPE_REV_INC >= 15).multiply(50000)).\
            multiply((DEBT <= 65)).multiply((GPS >= 19))


fundamantal_strategy = fundamantal_Strategy(data=data)
backtestEngine = Backtest(data.close, data.pct_chg, capital=1000000, tran_cost=0)
backtestEngine.caculate_result(signal=fundamantal_strategy.factor(), start_date="20200101", end_date="20200930", period=22, stock_num=10)