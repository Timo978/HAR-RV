# Required packages
import pandas as pd
import numpy as np
import warnings
import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm
# from datetime import datetime
from statsmodels.tsa.stattools import adfuller as adf
from statsmodels.graphics.gofplots import qqplot
from pandas.plotting import register_matplotlib_converters
from pandas.plotting import autocorrelation_plot
from pandas_datareader import data
from scipy import stats

warnings.filterwarnings('ignore')

start = datetime.datetime(2017,1,1)#获取数据的时间段-起始时间
end = datetime.date.today()#获取数据的时间段-结束时间
df = data.get_data_yahoo('BTC-USD', start=start, end=end)

df['log_ret'] = np.log(df['Close']/df['Close'].shift())
df['rv_daily'] = df['log_ret'].rolling(2).std() * np.sqrt(365)
df['rv_5'] = df['log_ret'].rolling(5).std() * np.sqrt(365)
df['rv_20'] = df['log_ret'].rolling(20).std() * np.sqrt(365)

df = df.dropna()

p25 = []
p50 = []
p75 = []
p100 = []
for i in range(21,len(df)):
    tmp = df.iloc[(i-21):i,:]
    tmp = tmp.sort_values(by='rv_daily')
    p2 = tmp.iloc[round(21*0.25),-3]
    p5 = tmp.iloc[round(21*0.50),-3]
    p7 = tmp.iloc[round(21*0.75),-3]
    p10 = tmp.iloc[round(21-1),-3]
    p25.append(p2)
    p50.append(p5)
    p75.append(p7)
    p100.append(p10)
plt.plot(p25)
plt.plot(p50)
plt.plot(p75)
plt.plot(p100)
plt.show()