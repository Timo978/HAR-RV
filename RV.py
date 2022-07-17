import pandas as pd
import numpy as np
import datetime
import warnings
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime
from statsmodels.tsa.stattools import adfuller as adf
from statsmodels.graphics.gofplots import qqplot
from pandas.plotting import autocorrelation_plot
from pandas_datareader import data
from scipy import stats

warnings.filterwarnings('ignore')

df = pd.DataFrame()
for i in ['2022-07-13','2022-07-14','2022-07-15']:
    tmp = pd.read_csv(f"C:/Users/tianm/Desktop/BTCUSDT-1min-{i}.csv")
    tmp.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Unknown']
    df = pd.concat([df,tmp],axis=0)

# df.columns = [['Date','Open','High','Low','Close','Volume','Unknown']]

df['Date'] = df.apply(lambda x:datetime.fromtimestamp(x['Date']),axis=1)
df.index = pd.to_datetime([x for x in df['Date'].squeeze().tolist()], dayfirst=True)
df.drop(["Date", "Unknown"], axis = 1, inplace = True)

sample = '5T'
df = df.resample(sample)
df = df.pad()
df.dropna(inplace = True)

df["D"] = df.index.date
n_periods = df.pivot_table(index = ["D"], aggfunc = 'size').values # aggfunc:类似于聚合函数，即往
df.columns = ['Open', 'High', 'Low', 'Close', 'Volume','D']

stats.describe(n_periods)

df.loc[df["D"] != df["D"].shift(), "Per"] = n_periods
df.fillna(method = 'ffill', inplace = True)

df["Ret_squared"] = np.where(df["D"] == df["D"].shift(),(np.log(df["Close"]/df["Close"].shift()))**2, np.nan)

# 日内RV
rv = df.groupby("D")["Ret_squared"].agg(np.sum).to_frame()

# Add sqrt to get Realized-Vol
rv.columns = ["RV_daily"]
rv["RV_daily"] = np.sqrt(rv["RV_daily"])

date = str(rv["RV_daily"].idxmax())
plt.plot(df["Close"].loc[date], label = f"{sample} close price")
plt.title(f"BTC/USDT on {date}")
plt.legend()
plt.show()

#
rv["RV_weekly"] = rv["RV_daily"].rolling(5).mean()
rv["RV_monthly"] = rv["RV_daily"].rolling(21).mean()
rv.dropna(inplace = True)
