# Portfolio-Risk-Analysis-and-Optimization

This project aims to analyze the risk and optimize portfolios using historical stock data. It includes calculations of risk metrics such as Value at Risk (VaR) and Conditional Value at Risk (CVaR), as well as the creation of different portfolios based on various weighting schemes.

## Introduction
Portfolio risk analysis plays a crucial role in investment decision-making. This project utilizes historical stock data to calculate risk metrics and construct portfolios with different weighting schemes. By analyzing risk characteristics and optimizing portfolio allocations, investors can make informed investment decisions.

## Dependencies
The following dependencies are required to run the project:

Python (version >= 3.7)
pandas
numpy
matplotlib
BeautifulSoup
pandas_datareader
scipy

## Installation
Clone the repository:

```bash
git clone https://github.com/jashbhatia07/portfolio-risk-analysis.git
```
Install the required dependencies using pip:


### Run the main script:
```python 
Risk Analysis and Optimization.py
```
The script will fetch historical stock data, calculate risk metrics, and construct portfolios based on different weighting schemes.

The results will be displayed in the console and saved as dataframes in the respective variables.

## Data Collection
The project fetches historical stock data from Yahoo Finance using the pandas_datareader library. The stock tickers and corresponding sectors are scraped from the Wikipedia page of S&P 500 companies using web scraping techniques with the urllib.request and BeautifulSoup libraries.

## Risk Metrics Calculation
The project calculates risk metrics such as Value at Risk (VaR) and Conditional Value at Risk (CVaR) using the variance-covariance approach. These metrics provide insights into the potential downside risk of the portfolio.

## Portfolio Construction
The project constructs portfolios based on different weighting schemes:

Tgcy Portfolio: Implements a specific weighting formula based on returns and covariances of the stocks.

Equal Weighted Portfolio: Assigns equal weights to each stock in the portfolio.

Risk Parity Weighted Portfolio: Allocates weights based on risk parity principles, considering volatilities and correlation between stocks.


## Results
The project provides the following results for each portfolio:

Portfolio Returns

Cumulative Returns

VaR (Value at Risk)

CVaR (Conditional Value at Risk)

Additional metrics for Risk Parity Weighted Portfolio: VaR with risk rebalancing (VaRR)

