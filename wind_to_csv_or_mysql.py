from WindPy import w
import pandas as pd
from sqlalchemy import create_engine
import pymysql
import re
import time
import matplotlib.pyplot as plt

conn = pymysql.connect(host='127.0.0.1',user='root', passwd="211411qwe", db='mysql')
cur = conn.cursor()
sql = 'CREATE DATABASE IF NOT EXISTS wind_wsd'
cur.execute(sql)
sql = 'Use wind_wsd;'
cur.execute(sql)

engine = create_engine("mysql+pymysql://root:211411qwe@127.0.0.1:3306/wind_wsd?charset=utf8")

# 初始化wind接口
print(w.start())
# print(w.isconnected())

# 获取股票列表
wdata = w.wset("listedsecuritygeneralview","sectorid=a001010100000000")
stock_codes = (wdata.Data)[0]
# print(stock_codes)

w_data = w.wsd(stock_codes, "mkt_cap_ard", "2010-01-01", "2011-01-01", "unit=1;Fill=Previous")
df = pd.DataFrame(w_data.Data, index=w_data.Fields, columns=w_data.Times)
df.to_csv("D:\PycharmProjects\mnist_PCA\Programming_Project\财务数据\总市值_20102011")
