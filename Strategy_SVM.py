from backtest import *
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from libsvm.python.svmutil import *
from libsvm.python.svm import *


class Data_SVM(Data):
    def __init__(self, path):
        Data.__init__(self, path)
        indexes = pd.read_csv(path + "index/标普500_中证全指.csv", encoding="gbk", index_col="date")
        self.CSI = indexes["000985.CSI"]
        self.SP500 = indexes["SPX.GI"]
        self.USDCNY = pd.read_csv(path + "exchange/USD_CNY历史数据.csv", index_col="date")["close"]


data_svm = Data_SVM(path="F:/Python/backtest_/")


class Strategy_SVM(Strategy):

    def pct_chg_PCA(self):
        # 构建指数中的股票主要成分的一日变化率，也就是pct_chg
        # 这里选择的指数用中证全指，就可以用全部股票的Pct_chg去算
        pct_chg = self.data.pct_chg
        # 填充NaN
        pct_chg = pct_chg.fillna(1)
        pca = PCA(20)
        new_pct_chg = pca.fit_transform(pct_chg)
        # 降维后的20只股票
        inv_pct_chg = pca.inverse_transform(new_pct_chg)
        inv_pct_chg = pd.DataFrame(inv_pct_chg, index=pct_chg.index)
        return inv_pct_chg

    def svm_predict_diretion(self):
        # 构造指数的收益率
        CSI_pct_chg = (self.data.CSI - self.data.CSI.shift(1))/self.data.CSI.shift(1)
        # 构造指数涨跌标签
        y = (CSI_pct_chg > 0).shift(-1)
        y.dropna(axis=0,how="all", inplace=True)

        # 标普500的三天收益率
        SP500_RDP_3 = 100 * (self.data.SP500 - self.data.SP500.shift(3))/self.data.SP500.shift(3)
        SP500_RDP_3.dropna(axis=0,how='all', inplace=True)
        # 构造汇率的三天收益率
        USDCNY_RDP_3 = 100 * (self.data.USDCNY - self.data.USDCNY.shift(3))/self.data.USDCNY.shift(3)
        USDCNY_RDP_3.dropna(axis=0, how='all', inplace=True)

        stocks_data = (self.pct_chg_PCA()).loc["20120101":"20191231"]
        SP500_RDP_3 = SP500_RDP_3.loc["20120101":"20191231"]
        USDCNY_RDP_3 = USDCNY_RDP_3[stocks_data.index]
        USDCNY_RDP_3 = USDCNY_RDP_3.loc["20120101":"20191231"]
        y = y.loc["20120101":"20191231"]
        # 合并全部特征
        x = pd.concat([stocks_data, SP500_RDP_3, USDCNY_RDP_3],axis=1)

        # 滚动利用svm进行预测，并计算预测准确率
        acc_list = []

        train_x = (x.loc["20120101":"20141231"]).as_matrix()
        train_y = (y.loc["20120101":"20141231"]).as_matrix()

        test_x = (x.loc["20150101":"20171231"]).as_matrix()
        test_y = (y.loc["20150101":"20171231"]).as_matrix()

        model = svm_train(train_y, train_x)
        p_label, p_acc, p_val = svm_predict(test_y, test_x, model)
        acc_list.append(p_acc)

        train_x = (x.loc["20130101":"20151231"]).as_matrix()
        train_y = (y.loc["20130101":"20151231"]).as_matrix()

        test_x = (x.loc["20160101":"20181231"]).as_matrix()
        test_y = (y.loc["20160101":"20181231"]).as_matrix()

        model = svm_train(train_y, train_x)
        p_label, p_acc, p_val = svm_predict(test_y, test_x, model)
        acc_list.append(p_acc)

        train_x = (x.loc["20140101":"20161231"]).as_matrix()
        train_y = (y.loc["20140101":"20161231"]).as_matrix()

        test_x = (x.loc["20170101":"20191231"]).as_matrix()
        test_y = (y.loc["20170101":"20191231"]).as_matrix()

        model = svm_train(train_y, train_x)
        p_label, p_acc, p_val = svm_predict(test_y, test_x, model)
        acc_list.append(p_acc)

        print(acc_list)



strategy_svm = Strategy_SVM(data_svm)
pct_chg_pca = strategy_svm.pct_chg_PCA()
strategy_svm.svm_predict_diretion()
