import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime
from statsmodels.tsa.stattools import adfuller as adf
from statsmodels.graphics.gofplots import qqplot
from pandas.plotting import register_matplotlib_converters
from pandas.plotting import autocorrelation_plot
from pandas_datareader import data
from scipy import stats

warnings.filterwarnings('ignore')

ticker = "SPY"

# Import & clean intraday data
df = pd.read_csv(f"{ticker}1min_clean.csv")

df.rename(columns = {"Unnamed: 0":"Date"}, inplace = True)
df.index = pd.to_datetime(df["Date"])
df.drop(["Date", "Inc Vol"], axis = 1, inplace = True)

df.tail()

# Resample 1min to 5min
sample = '5T'
df = df.resample(sample).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
df.dropna(inplace = True)

df.tail()

# Download VIX & SPY volume data from yahoo finance
start = str(df.index[0].date())
end = str(df.index[-1].date())

SPY_daily = data.DataReader("SPY", start=start, end=end, data_source="yahoo")[["Volume"]]# Import only the columns we need
VIX_daily = data.DataReader("^VIX", start=start, end=end, data_source="yahoo")[["Adj Close"]]

SPY_daily.plot(title = "SPY Volume");plt.show()
VIX_daily.plot(title = "VIX Adj Close");plt.show()

# Compute number of periods/day
df["D"] = df.index.date
n_periods = df.pivot_table(index = ["D"], aggfunc = 'size').values

stats.describe(n_periods)

df.loc[df["D"] != df["D"].shift(), "Per"]  = n_periods
df.fillna(method = 'ffill', inplace = True)

df["Ret"] = np.where(df["D"] == df["D"].shift(),
                    ( (df["Close"]-df["Close"].shift()) * 1/df["Per"] ) **2, np.nan)

# Perform the sum grouped by days.
rv = df.groupby("D")["Ret"].agg(np.sum).to_frame()

# Add sqrt to get Realized-Vol
rv.columns = ["RV_daily"]
rv["RV_daily"] = np.sqrt(rv["RV_daily"])

# Check what day in the dataset had the highest realized vol
date = str(rv["RV_daily"].idxmax())
plt.plot(df["Close"].loc[date], label = f"{sample} close price")
plt.title(f"{ticker} on {date}")
plt.legend()
plt.show()

# Compute weekly and monthly RV.
rv["RV_weekly"] = rv["RV_daily"].rolling(5).mean()
rv["RV_monthly"] = rv["RV_daily"].rolling(21).mean()
rv.dropna(inplace = True)

#Add IV & Volume variables
rv["VIX"] = VIX_daily.loc[rv.index]
rv["SPY_volume"] = SPY_daily.loc[rv.index]

print(rv.head()); print(rv.describe())

# Check for stationarity with adf test
print("p-value for daily RV:", adf(rv["RV_daily"].values)[1])
print("p-value for weekly RV:",adf(rv["RV_weekly"].values)[1])
print("p-value for monthly RV:",adf(rv["RV_monthly"].values)[1])
print("p-value for VIX Adj Close:",adf(rv["VIX"].values)[1])
print("p-value for SPY Volume:",adf(rv["SPY_volume"].values)[1])

# Plot the RV variables.
rv[["RV_daily","RV_weekly","RV_monthly"]].plot(title = f"{ticker} realized volatility from {df.index.date[0]} to {df.index.date[-1]}")
plt.ylabel("RV")
plt.show()

# Prepare data
rv["Target"] = rv["RV_daily"].shift(-1) #We want to predict the RV of the next day.
rv.dropna(inplace = True)

#Scale the data
rv_scaled = (rv-rv.min())/(rv.max()-rv.min())

#Add constant c
rv_scaled = sm.add_constant(rv_scaled)

#Split train and test sets
split = int(0.60 * rv.shape[0])
X = rv_scaled.drop("Target", axis = 1)
y = rv_scaled[["Target"]]
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

results = sm.OLS(y_train, X_train).fit()
results.summary()

# Perform out of sample prediction
y_hat = results.predict(X_test)

plt.figure(figsize = (14,5))

#Predicted RV
plt.subplot(1,2,1)
plt.plot(y_test.index, y_hat)
plt.title("Predicted RV")

#Actual RV
plt.subplot(1,2,2)
plt.plot(y_test.index, y_test, color = "orange")
plt.title("Actual RV")
plt.show()

plt.scatter(y_test, y_hat)
plt.title("Predicted vs observed RV")
plt.show()

# Metrics
def score(y_hat, y, metric):
    """Return metrics of y_hat vs. y

    Args:
        y_hat (np.array): Predicted values
        y (np.array): Actual values
        metric (str): Metric to use

    Returns:
        float: The metric
    """
    if metric == "MSE":
        return np.mean( (y_hat-y)**2)
    elif metric == "R_squared":
        ss_res = np.sum( (y - y_hat)**2 )
        ss_tot = np.sum( (y - np.average(y)) **2)
        return 1 - ss_res/ss_tot
    elif metric == "MAE":
        return np.mean( np.abs(y-y_hat))


# In-sample scores
print("In-sample scores")

y_hat_is = results.predict(X_train)
mse_is = score(y_hat_is, y_train.values.ravel(), "MSE")
r_sq_is = score(y_hat_is, y_train.values.ravel(), "R_squared")
mae_is = score(y_hat_is, y_train.values.ravel(), "MAE")

print(f"MSE:{mse_is}, R^2:{r_sq_is}, MAE:{mae_is}")

print("----------------")

# Out-of-sample scores
print("Out-of-sample scores")

mse_oos = score(y_hat, y_test.values.ravel(), "MSE")
r_sq_oos = score(y_hat, y_test.values.ravel(), "R_squared")
mae_oos = score(y_hat, y_test.values.ravel(), "MAE")

print(f"MSE:{mse_oos}, R^2:{r_sq_oos}, MAE:{mae_oos}")

# Residuals
residuals = y_test.values.ravel() - y_hat

autocorrelation_plot(residuals);plt.show()
qqplot(residuals);plt.show()
plt.plot(residuals);plt.show()
plt.hist(residuals, bins = 50); plt.show()