from openbb_terminal.sdk import openbb

import datetime
import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import tqdm
import warnings
warnings.filterwarnings("ignore")

def ticker_to_yahoo_format(t):
    return '.'.join(t.split(':')[::-1])[:-1]

@st.cache()
def get_daily_data_with_today(ticker, today, start_date=None, n_days=None, enforce_today=True, adj_div=True):
    if ticker[0] == 'H':
        ticker = ticker_to_yahoo_format(ticker)
    if start_date is None:
        start_date = today - (datetime.timedelta(days=n_days if n_days is not None else 365))
    data_until_today = openbb.stocks.load(
        ticker,
        interval=1440,
        start_date=start_date,
        verbose=False
    ).drop('Adj Close', axis=1)
    
    if enforce_today:
        data_today = openbb.stocks.load(
            ticker,
            interval=1,
            start_date=today - datetime.timedelta(days=1),
            end_date=today,
            verbose=False
        ).drop('Adj Close', axis=1)
        data_today = data_today[data_today.index.time != datetime.time(16, 8)]
        data_today = data_today.groupby(data_today.index.floor('1D')).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
        data_today = data_today[data_today.index.date == today.date()]
        if not len(data_today):
            raise ValueError('No minute data for today')
        data = pd.concat([data_until_today[data_until_today.index < today], data_today])
    else:
        data = data_until_today
    
    if adj_div:
        div_data = openbb.stocks.fa.divs(ticker)
        if div_data.empty:
            div_dct = {}
        else:
            div_data.drop('Change', axis=1, inplace=True)
            div_data.index = div_data.index.floor('1D').tz_localize(None)
            div_data['prev_Date'] = div_data.index - datetime.timedelta(days=1)
            merged_data = pd.merge_asof(div_data.sort_index().reset_index(), data[['Close']].reset_index(),
                                        direction='backward', left_on='prev_Date', right_on='date').dropna()
            merged_data['yield'] = merged_data['Dividends'] / merged_data['Close']
            if len(merged_data):
                div_dct = merged_data[['date', 'yield']].set_index('date')['yield'].to_dict()
            else:
                div_dct = {}
        for ex_date, yield_rate in sorted(div_dct.items()):
            data.loc[data.index <= ex_date, ['Open', 'High', 'Low', 'Close']] *= (1 - yield_rate)
    return data

def SMMA(s, n):
#     никто все равно так не считает, все берут ЕМА обычный
#     а раз все берут ЕМА, то и "самосбывающееся пророчество" по этому индикатору происходит по ЕМА
#     return s.rolling(n).mean().dropna().ewm(alpha=1/n).mean()
    return s.ewm(alpha=1/n).mean()

def RSI(data, n=14):
    d = data['Close']
    U = SMMA(np.maximum(data['Close'] - data['Close'].shift(1), 0).dropna(), n)
    D = SMMA(np.maximum(data['Close'].shift(1) - data['Close'], 0).dropna(), n)
    return 100 * U / (U + D + 1e-25)

def draw_rsi(ticker_list, figsize=(10, 7)):
    rsi_data = {}
    for ticker in ((ticker_list).split('\n')):
        data = get_daily_data_with_today(ticker, datetime.datetime.combine(datetime.date.today(), datetime.time()), enforce_today=True)
        rsi_data[ticker] = RSI(data)
    rsi_data = pd.DataFrame(rsi_data)
    rsi_data.index = [str(x)[-5:].replace('-', '/') for x in rsi_data.index.date]
    rsi_data.columns = [x[4:] for x in rsi_data.columns]

    plt.figure(figsize=figsize)
    ax = sns.heatmap(rsi_data.T.iloc[:, -14:].sort_values(rsi_data.index[-1], ascending=False), annot=True, cbar=False)
    ax.tick_params(left=True, labelleft=True, right=True, labelright=True, rotation=0)
    plt.title('RSI')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=200)
    buf.seek(0)
    plt.close()
    return buf