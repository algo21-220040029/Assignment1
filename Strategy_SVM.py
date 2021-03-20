from backtest import *
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

class Data_SVM(Data):
    def __init__(self, path):
        Data.__init__(self, path)
        indexes = pd.read_csv("index/标普500_中证全指.csv", encoding="gbk", index_col="date")
        self.CSI = indexes["000985.CSI"]
        self.SP500 = indexes["SPX.GI"]

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
        return inv_pct_chg

    def svm_predict_diretion(self):
        train_y = (self.data.CSI > 0).shift(-1)
        print(train_y)


strategy_svm = Strategy_SVM(data_svm)
pct_chg_pca = strategy_svm.pct_chg_PCA()
strategy_svm.svm_predict_diretion()
