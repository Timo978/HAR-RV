import numpy as np
import pandas as pd
import datetime
from pandas_datareader import data
import matplotlib.pyplot as plt

start = datetime.datetime(2017,1,1)#获取数据的时间段-起始时间
end = datetime.date.today()#获取数据的时间段-结束时间
df = data.get_data_yahoo(['BTC-USD'], start=start, end=end)

df['log_ret'] = np.log(df['Close']/df['Close'].shift())
df['rv_5'] = df['log_ret'].rolling(5).std() * np.sqrt(365)
df['rv_20'] = df['log_ret'].rolling(20).std() * np.sqrt(365)

