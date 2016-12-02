import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

def load_data():
    web_traf = pd.read_csv('data/input01.txt')

    # logins.rename(columns={0:'time'}, inplace=True)
    # logins['count'] = 1
    # logins.set_index(pd.to_datetime(logins['time']), inplace=True)
    # logins.drop('time', axis=1, inplace=True)
    #
    # hourly = logins.resample(rule = 'h').count()
    # daily = logins.resample(rule = 'd').count()
    #
    return web_traf

def df_plot(df):
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(1,1,1)
    ax.plot(df)
    plt.show()

def weekend_plot(daily):
    # daily['dayofweek'] = daily.index.dayofweek
    daily['dayofweek'] = daily.index
    daily['weekend'] = np.where(((daily['dayofweek'].values == 5) | (daily['dayofweek'].values == 6)), 1, 0)

    weekend = daily[daily['weekend'] == 1]
    print weekend

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(daily.index , daily)
    for i in xrange(0, weekend.shape[0], 2):
        ax.fill_between(weekend.index[i:i+2], 0, weekend['500'][i:i+2])
    plt.show()

def plot_acf_pacf(your_data, lags):
   fig = plt.figure(figsize=(12,8))
   ax1 = fig.add_subplot(211)
   fig = plot_acf(your_data, lags=lags, ax=ax1)
   ax2 = fig.add_subplot(212)
   fig = plot_pacf(your_data, lags=lags, ax=ax2)
   plt.show()

def mod(df, p, d, q, P, D, Q, lag):
    # print df
    model = SARIMAX(endog=df['Y'].values, order=(p, d, q), seasonal_order=(P, D, Q, lag)).fit()
    print model.summary()

    plt.plot(df.index, model.resid); plt.show()
    plot_acf_pacf(model.resid[7:], 21)

def mod_fit(df, p, d, q, P, D, Q, lag):
    model = SARIMAX(endog=df['Y'].values, order=(p, d, q), seasonal_order=(P, D, Q, lag)).fit()
    return model


if __name__ == "__main__":
    daily = load_data()

    # df_plot(daily)
    # weekend_plot(daily) #isn't working...don't want to make it work now

# # box-jenkins methodology
    ## differenced and seasonally differenced
    df = pd.concat([daily, daily.shift(), daily.shift(7), daily.shift(8)], axis=1).dropna()
    ## alternatively
    # daily.diff(periods=1) #gives the difference between consecutive rows
    # daily.diff(periods=7) #gives the difference between rows 7 apart
    df.columns = ['Y', 'LY', 'L7Y', 'L8Y']
    # df['SARIMA000010'] = df.Y - df.L7Y
    # df['SARIMA010010'] = (df.Y - df.L7Y) - (df.LY - df.L8Y)

    ## plots of each of these
    # df_plot(df['SARIMA000010']) #absolutely not stationary
    # print df['SARIMA000010'].std()
    # df_plot(df['SARIMA010010']) #looks more stationary but possibly over-differenced since series alternates between positive and negative...also perhaps needs a constant
    # print df['SARIMA010010'].std()

    ## let's work with the latter (0,1,0)X(0,1,0) first
    # plot_acf_pacf(df['SARIMA010010'], 28) #looks like I want to add an SMA maybe term...since the first lag is negative and (nearly or barely) significant and L7, L14, etc. are too

    # mod(df, 0, 1, 1, 0, 1, 1, 7) #it looks like the residuals might be increasing in variance over time...try taking the log transform

    df.Y = df.Y.apply(lambda x: np.log(x))
    # mod(df, 0, 1, 1, 0, 1, 1, 7) #wow...I think it looks pretty good.

    expected = pd.read_csv('data/output01.txt', header=None)
    model = mod_fit(df, 0, 1, 1, 0, 1, 1, 7) #wow...I think it looks pretty good.
    # print model.get_forecast(30)
    # print 100*np.sum(np.abs(np.exp(model.forecast(30)) - expected.values.T[0])/expected.values.T[0])
    print np.exp(model.forecast(30)), expected.values.T[0]
    # print expected.values.T[0]






## this is the code I would submit...I can't use the sarimax module, unfortunately.
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# import pandas as pd
# import numpy as np
#
# N = int(raw_input())
# data = []
# for i in xrange(N):
#     data.append(int(raw_input()))
# df = pd.DataFrame(data, columns=["Y"])
#
# df.Y = df.Y.apply(lambda x: np.log(x))
#
# model = SARIMAX(endog=df['Y'].values, order=(0, 1, 1), seasonal_order=(0, 1, 1, 7)).fit()
# print np.exp(model.forecast(30))
