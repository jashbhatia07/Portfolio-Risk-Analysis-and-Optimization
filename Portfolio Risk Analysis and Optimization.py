import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup as bs
from pandas_datareader import data
from datetime import date
from scipy.stats import norm
from pandas_datareader._utils import RemoteDataError
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()

site = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
page = urllib.request.urlopen(site)
soup = bs(page.read(), "html5lib")  
table = soup.find('table', {'class': 'wikitable'})  
'''
SD = dict()  
for row in table.findAll('tr'):  # 'tr'是HTML语言中的行
    col = row.findAll('td')  # 'td'是HTML中的列
    if len(col) > 0:
        ticker = str(col[0].string.strip())
        sector = str(col[3].string.strip()).lower()
        SD[ticker] = sector
'''
SD = dict() 
for row in table.findAll('tr'):  
    col = row.findAll('td') 
    if len(col) > 0:
        ticker = str(col[0].string.strip()) if col[0].string else ""
        sector = str(col[3].string.strip()).lower() if col[3].string else ""
        SD[ticker] = sector


SP500 = pd.Series(SD)  
SP500.to_csv('SP500 tickers.csv')

start = date(2010, 1, 1)
end = date(2020, 1, 1)


SP500_Tickers = []
SP500_csv = pd.read_csv('SP500 tickers.csv', header=None)
for item in SP500_csv[0]:
    SP500_Tickers.append(item)



AAPL = pdr.get_data_yahoo("AAPL", start=start, end=end)
GOOG = pdr.get_data_yahoo("GOOG", start=start, end=end)
SPY = pdr.get_data_yahoo("SPY", start=start, end=end)


def VaR(value, CI, mu, sigma):
    return (norm.ppf(CI)*sigma-mu) * value  # VARIANCE_COVARIANCE APPROACH


def CVaR(value, CI, mu, sigma):
    return ((1-CI)**-1 * norm.pdf(norm.ppf(1-CI))*sigma-mu) * value
#  http://www.quantatrisk.com/2016/12/08/conditional-value-at-risk-normal-student-t-var-model-python/


AAPL['log_ret'] = np.log(AAPL['Adj Close'] / AAPL['Adj Close'].shift(1))
GOOG['log_ret'] = np.log(GOOG['Adj Close'] / GOOG['Adj Close'].shift(1))
AAPL['mov_mean'] = AAPL['log_ret'].rolling(window=252).mean()
GOOG['mov_mean'] = GOOG['log_ret'].rolling(window=252).mean()
AAPL['mov_vol'] = AAPL['log_ret'].rolling(window=252).std()
GOOG['mov_vol'] = GOOG['log_ret'].rolling(window=252).std()
AAPL['VaR'] = VaR(1, 0.95, AAPL['mov_mean'], AAPL['mov_vol'])
GOOG['VaR'] = VaR(1, 0.95, GOOG['mov_mean'], GOOG['mov_vol'])
Mov_Cov = AAPL['log_ret'].rolling(window=252).cov(GOOG['log_ret'])
Mov_Corr = AAPL['log_ret'].rolling(window=252).corr(GOOG['log_ret'])
AAPL['mov_var'] = AAPL['log_ret'].rolling(window=252).var()
GOOG['mov_var'] = GOOG['log_ret'].rolling(window=252).var()

Rf = 0.02 / 252  # risk free rate

# Tgcy_portfolio  https://business.missouri.edu/yanx/fin333/lectures/Riskyportfolio%20short.pdf
Tgcy_wgt_AAPL = ((AAPL['log_ret'] - Rf) * GOOG['mov_var'] - (GOOG['log_ret'] - Rf) * Mov_Cov) / \
                ((AAPL['log_ret'] - Rf) * GOOG['mov_var'] - (GOOG['log_ret'] - Rf) * AAPL['mov_var'] - \
                 (AAPL['log_ret'] - Rf + GOOG['log_ret'] - Rf) * Mov_Cov)  # 计算两个资产权重的公式

Tgcy_wgt_AAPL[Tgcy_wgt_AAPL<0] = 0   # assume long only
Tgcy_wgt_AAPL[Tgcy_wgt_AAPL>1] = 1
Tgcy_wgt_GOOG = 1 - Tgcy_wgt_AAPL
Tgcy_portfolio_ret = AAPL['log_ret'] * Tgcy_wgt_AAPL + GOOG['log_ret'] * Tgcy_wgt_GOOG

Tgcy_portfolio_mean = Tgcy_portfolio_ret.rolling(window=252).mean()
Tgcy_portfolio_std = Tgcy_portfolio_ret.rolling(window=252).std()
Tgcy_portfolio_VaR = VaR(1, 0.95, Tgcy_portfolio_mean, Tgcy_portfolio_std)
Tgcy_portfolio_CVaR = CVaR(1, 0.95, Tgcy_portfolio_mean, Tgcy_portfolio_std)
Tgcy_portfolio = pd.DataFrame({
    'Return': Tgcy_portfolio_ret,
    'VaR': Tgcy_portfolio_VaR,
    'CVaR': Tgcy_portfolio_CVaR,
    'Cum_Return': Tgcy_portfolio_ret.cumsum()
})
Tgcy_portfolio = Tgcy_portfolio[['Return', 'Cum_Return', 'VaR', 'CVaR']]  # Reorder columns in the desired sequence


# # # Equal weighted portfolio
Eql_portfolio_ret = AAPL['log_ret'] * 0.5 + GOOG['log_ret'] * 0.5

Eql_portfolio_mean = Eql_portfolio_ret.rolling(window=252).mean()
Eql_portfolio_std = Eql_portfolio_ret.rolling(window=252).std()
Eql_portfolio_VaR = VaR(1, 0.95, Eql_portfolio_mean, Eql_portfolio_std)
Eql_portfolio_CVaR = CVaR(1, 0.95, Eql_portfolio_mean, Eql_portfolio_std)
Eql_portfolio = pd.DataFrame({
    'Return': Eql_portfolio_ret,
    'VaR': Eql_portfolio_VaR,
    'CVaR': Eql_portfolio_CVaR,
    'Cum_Return': Eql_portfolio_ret.cumsum()
})
Eql_portfolio = Eql_portfolio[['Return', 'Cum_Return', 'VaR', 'CVaR']]  # Reorder columns in the desired sequence

# # # Risk parity weighted portfolio
# article http://people.umass.edu/kazemi/An%20Introduction%20to%20Risk%20Parity.pdf
RskPrty_wgt_AAPL = GOOG['mov_vol'] / (GOOG['mov_vol']+AAPL['mov_vol'])
RskPrty_wgt_GOOG = 1 - RskPrty_wgt_AAPL
RskPrty_portfolio_ret = AAPL['log_ret'] * RskPrty_wgt_AAPL + GOOG['log_ret'] * RskPrty_wgt_GOOG

RskPrty_portfolio_mean = RskPrty_portfolio_ret.rolling(window=252).mean()
RskPrty_portfolio_std = RskPrty_portfolio_ret.rolling(window=252).std()
RskPrty_portfolio_VaR = VaR(1, 0.95, RskPrty_portfolio_mean, RskPrty_portfolio_std)

RskPrty_portfolio_VaRR = np.sqrt(RskPrty_wgt_AAPL**2 * AAPL['VaR']**2 + RskPrty_wgt_GOOG**2 * GOOG['VaR']**2 \
                                 +2*RskPrty_wgt_AAPL*RskPrty_wgt_GOOG*AAPL['VaR']*GOOG['VaR']\
                                 *Mov_Corr)
RskPrty_portfolio_CVaR = CVaR(1, 0.95, RskPrty_portfolio_mean, Eql_portfolio_std)
RskPrty_portfolio = pd.DataFrame({'Return': RskPrty_portfolio_ret,
                              'VaR': RskPrty_portfolio_VaR,
                              'VaRR': RskPrty_portfolio_VaRR,
                              'CVaR': RskPrty_portfolio_CVaR,
                              'Cum_Return': RskPrty_portfolio_ret.cumsum()})
RskPrty_portfolio = RskPrty_portfolio[['Return', 'Cum_Return', 'VaR', 'VaRR', 'CVaR']]

print(RskPrty_portfolio)



