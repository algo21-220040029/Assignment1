import pandas as pd
import numpy as np
import pymysql
from sqlalchemy import create_engine
import re
import tushare as ts
import time
import baostock as bs
import matplotlib.pyplot as plt

conn = pymysql.connect(host='127.0.0.1',user='root', passwd="211411qwe", db='mysql')
cur = conn.cursor()
sql = 'CREATE DATABASE IF NOT EXISTS baostock_hfq'
cur.execute(sql)
sql = 'Use baostock_hfq;'
cur.execute(sql)

engine = create_engine("mysql+pymysql://root:211411qwe@127.0.0.1:3306/baostock_hfq?charset=utf8")

ts.set_token('e3fcccabcf19b3c8e410009be322617a8d944cdee5906e6f8032683f')
pro = ts.pro_api()

lg = bs.login()

def ts_stock_basic_mysql(pro,cur,engine):
    data = pro.stock_basic(list_status = 'L',fields='ts_code,symbol,name,area,industry,market,list_date')

    sql = "CREATE TABLE IF NOT EXISTS STOCK_BASIC(ts_code CHAR(10) PRIMARY KEY,symbol CHAR(7),name CHAR(15),area CHAR(7),industry CHAR(15),market CHAR(7), list_date CHAR(10))" \
          "ENGINE=innodb DEFAULT CHARSET=utf8"
    cur.execute(sql)
    data.to_sql('stock_basic', engine, if_exists='replace')
    data = pro.stock_basic(list_status = 'D',fields='ts_code,symbol,name,area,industry,market,list_date')
    data.to_sql('stock_basic', engine, if_exists='append')
    data = pro.stock_basic(list_status = 'P',fields='ts_code,symbol,name,area,industry,market,list_date')
    data.to_sql('stock_basic', engine, if_exists='append')

def bs_daily_mysql(pro,cur,engine):
    sql = 'select ts_code from stock_basic'
    cur.execute(sql)
    ts_codes = str(cur.fetchall())
    # 处理从数据库提取的关于股票代码的数据
    ts_code_list = re.findall("(\'.*?\')", ts_codes)
    ts_code_list = [re.sub("'", '', each) for each in ts_code_list]
    print(ts_code_list)
    bs_code_list = []
    for i in range(len(ts_code_list)):
        stock_code = ts_code_list[i][0:6]
        # print(stock_code)
        ex = (ts_code_list[i][-2:]).lower()
        # print(ex)
        bs_code = ex+'.'+stock_code
        bs_code_list.append(bs_code)
    # print(bs_code_list)

    sql = "show tables;"
    cur.execute(sql)
    tables = [cur.fetchall()]
    table_list = re.findall('(\'.*?\')', str(tables))
    table_list = [re.sub("'", '', each) for each in table_list]
    # print(table_list)

    sql = "select table_name, table_rows from information_schema.tables where table_schema = 'baostock' and table_rows <1;"
    cur.execute(sql)
    empty_tables = [cur.fetchall()]
    empty_table_list = re.findall('(\'.*?\')', str(empty_tables))
    empty_table_list = [re.sub("'", '', each) for each in empty_table_list]
    # print(empty_table_list)

    for bs_code in bs_code_list:
        # print(bs_code)
        name = (bs_code.replace('.', '_') + "_daily").lower()
        if name in table_list and name not in empty_table_list:
            print(bs_code + "对应的日线表已存在且不为空")
            continue
        sql = "CREATE TABLE IF NOT EXISTS "+bs_code.replace('.','_') +"_daily(date CHAR(11)," \
              "code CHAR(10) PRIMARY KEY,open float(4,2),high float(4,2),low float(4,2),close float(4,2), preclose float(4,2)," \
              "volume float(11,2),amount float(11,2),turn float(11,2),tradestatus float(11,2), pctChg float(11,2),peTTM float(11,2),psTTM float(11,2),pcfNcfTTM float(11,2)," \
                                                                      "pbMRQ float(11,2),isST INT UNSIGNED) ENGINE=innodb DEFAULT CHARSET=utf8"
        # print(bs_code.replace('.','_') +"_daily")
        cur.execute(sql)
        rs = bs.query_history_k_data_plus(bs_code,
                                          "date,code,open,high,low,close,preclose,volume,amount,"
                                          "turn,tradestatus,pctChg,peTTM,psTTM,pcfNcfTTM,pbMRQ,isST",
                                          start_date='2010-01-01', end_date='2020-11-15',
                                          frequency="d", adjustflag="1")
        data_list = []
        while (rs.error_code == '0') & rs.next():
            # 获取一条记录，将记录合并在一起
            data_list.append(rs.get_row_data())
        result = pd.DataFrame(data_list, columns=rs.fields)
        date = result['date']
        for index in range(0, len(date)):
            new_date = str(date[index]).replace("-", "")
            date[index] = new_date
        if (result is None)or(result.empty):
            print(bs_code+"返回数据是空的，不导入数据")
        else:
            print(bs_code + "返回数据不为空，导入数据")
            result.to_sql(str(bs_code.replace('.','_')) + '_daily', engine, if_exists='replace')
        # print(str(bs_code.replace('.','_')) + '_daily')

def generate_data_of_allstocks(data_type):
    sql = 'select ts_code from stock_basic'
    cur.execute(sql)
    ts_codes = str(cur.fetchall())
    # 处理从数据库提取的关于股票代码的数据
    ts_code_list = re.findall("(\'.*?\')", ts_codes)
    ts_code_list = [re.sub("'", '', each) for each in ts_code_list]
    print(ts_code_list)
    bs_code_list = []
    for i in range(len(ts_code_list)):
        stock_code = ts_code_list[i][0:6]
        # print(stock_code)
        ex = (ts_code_list[i][-2:]).lower()
        # print(ex)
        bs_code = ex + '_' + stock_code+"_daily"
        # print(bs_code)
        bs_code_list.append(bs_code)
    for i in bs_code_list:
        sql = "select date," + data_type + " from " + i
        # print(sql)
        cur.execute(sql)
        data = cur.fetchall()
        if i == "sz_000001_daily":
            frame = pd.DataFrame(list(data))
            frame.rename(columns={0: 'date', 1: '000001'}, inplace=True)
        else:
            it_frame = pd.DataFrame(list(data))
            i = i[3:9]
            print(i)
            if it_frame.empty:
                print("股票"+i+"数据为空")
                continue
            it_frame.rename(columns={0: 'date', 1: i}, inplace=True)
            frame = pd.merge(frame, it_frame, on='date', how='left')
            # print(it_frame)
    frame.to_csv("hfq/"+data_type+"_of_all_stocks.csv",encoding="utf-8")
    # frame = pd.DataFrame(frame.values.T, index=frame.columns, columns=frame.index)
    # print(frame)
    # frame.to_sql(data_type+'_of_all_stocks', engine, if_exists='replace')


# ts_stock_basic_mysql(pro,cur,engine)
# bs_daily_mysql(pro,cur,engine)
# generate_data_of_allstocks("close")
# generate_data_of_allstocks("open")
# generate_data_of_allstocks("high")
# generate_data_of_allstocks("low")
# generate_data_of_allstocks("volume")
# generate_data_of_allstocks("amount")
# generate_data_of_allstocks("tradestatus")
# generate_data_of_allstocks("isST")
# generate_data_of_allstocks("turn")
# generate_data_of_allstocks("pctChg")
# generate_data_of_allstocks("peTTM")
# generate_data_of_allstocks("psTTM")
# generate_data_of_allstocks("pcfNcfTTM")
# generate_data_of_allstocks("pbMRQ")
