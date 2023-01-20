from tinkoff.invest import CandleInterval, Client
from tinkoff.invest.caching.market_data_cache.cache_settings import (
    MarketDataCacheSettings,
)
from tinkoff.invest.services import MarketDataCache
from tinkoff.invest.utils import now
from tinkoff.invest import CandleInterval, Client
from tinkoff.invest.caching.market_data_cache.cache_settings import (
    MarketDataCacheSettings,
)
from tinkoff.invest.services import MarketDataCache
from tinkoff.invest.utils import now

with open('./tcs_token.txt', 'r') as inf:
    TOKEN = inf.read().strip()

from openbb_terminal.sdk import openbb

from dataclasses import dataclass, asdict
from pathlib import Path
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

# @st.cache(ttl=24*3600)
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

def q2n(q):
    return (q.units + q.nano * 10**-9) if not isinstance(q, dict) else (q['units'] + q['nano'] * 10**-9)

def candle_list_to_df(candles, remove_incomplete=True):
    df = pd.DataFrame(
        [[
            c.time,
            q2n(c.open),
            q2n(c.high),
            q2n(c.low),
            q2n(c.close),
            c.volume
         ] for c in candles if (not remove_incomplete or c.is_complete)],
        columns=['date', 'Open', 'High', 'Low', 'Close', 'Volume']
    )
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None).dt.floor('1D')
    df.set_index('date', inplace=True)
    return df[df.index.weekday < 5].groupby(level=0).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})

def load_div_data(ticker, from_):
    with Client(TOKEN) as client:
        r = client.instruments.find_instrument(query=ticker)
        ok_instruments = []
        for i in r.instruments:
            if i.instrument_type == 'share' and i.class_code == 'TQBR' and i.ticker == ticker:
                ok_instruments.append(i)
        if len(ok_instruments) != 1:
            raise ValueError(f'Error mapping a ticker to figii, {len(ok_instruments)}')
        else:
            figi = ok_instruments[0].figi
            tink_inst = ok_instruments[0]

        df = pd.DataFrame([asdict(x) for x in client.instruments.get_dividends(figi=figi, from_=from_, to=now()).dividends])
        if df.empty:
            return {}
    
    df['yield_value'] = df['yield_value'].apply(q2n)
    df['last_buy_date'] = df['last_buy_date'].dt.tz_localize(None).dt.floor('1D')
    df = df[(pd.to_datetime(df.payment_date).dt.tz_localize(None) > pd.to_datetime(0)) &
                           (df.dividend_type != 'Cancelled')]
    df = df[['yield_value', 'last_buy_date']].groupby('last_buy_date')['yield_value'].sum()
    return df.to_dict()
    
# @st.cache(ttl=24*3600)
def load_tcs_data(ticker, from_, adj_div=True, enforce_today=False, remove_incomplete=True):
    with Client(TOKEN) as client:
        r = client.instruments.find_instrument(query=ticker)
        figis = []
        for i in r.instruments:
            if i.instrument_type == 'share' and i.class_code == 'TQBR' and i.ticker == ticker:
                figis.append(i.figi)
        if len(figis) != 1:
            raise ValueError(f'Error mapping a ticker to figi, {len(figis)}')
        else:
            figi = figis[0]
        
        settings = MarketDataCacheSettings(base_cache_dir=Path("market_data_cache1"))
        market_data_cache = MarketDataCache(settings=settings, services=client)
        
        candles = []
        for candle in market_data_cache.get_all_candles(
            figi=figi,
            from_=from_,
            interval=CandleInterval.CANDLE_INTERVAL_DAY,
        ):
            candles.append(candle)
        candle_df = candle_list_to_df(candles, remove_incomplete)
        if enforce_today:
            for candle in market_data_cache.get_all_candles(
                figi=figi,
                from_=now() - max(datetime.timedelta(days=7), datetime.datetime.now() - candle_df.index.max()),
                interval=CandleInterval.CANDLE_INTERVAL_1_MIN,
            ):
                candles.append(candle)
            new_candle_df = candle_list_to_df(candles, remove_incomplete)
            candle_df = pd.concat([
                candle_df,
                new_candle_df[~(new_candle_df.index.isin(candle_df.index))
                              & (new_candle_df.index < new_candle_df.index.max())]
            ]).sort_index()
        
        if adj_div:
            df = pd.DataFrame([asdict(x) for x in client.instruments.get_dividends(figi=figi, from_=from_, to=now()).dividends])
            if df.empty:
                div_dct = {}
            else:
                df['yield_value'] = df['yield_value'].apply(q2n)
                df['last_buy_date'] = df['last_buy_date'].dt.tz_localize(None).dt.floor('1D')
                df = df[(pd.to_datetime(df.payment_date).dt.tz_localize(None) > pd.to_datetime(0)) &
                                       (df.dividend_type != 'Cancelled')]
                df = df[['yield_value', 'last_buy_date']].groupby('last_buy_date')['yield_value'].sum()
                div_dct = df.to_dict()
            for ex_date, yield_rate in sorted(div_dct.items()):
                candle_df.loc[candle_df.index <= ex_date, ['Open', 'High', 'Low', 'Close']] *= (1 - yield_rate / 100)
    return candle_df

def SMMA(s, n):
#     никто все равно так не считает, все берут ЕМА обычный
#     а раз все берут ЕМА, то и "самосбывающееся пророчество" по этому индикатору происходит по ЕМА
#     return s.rolling(n).mean().dropna().ewm(alpha=1/n).mean()
    return pd.Series(s).ewm(alpha=1/n).mean()

def EMA(s, n):
    return pd.Series(s).ewm(span=n).mean()

def MA(s, n):
    return pd.Series(s).rolling(n, 1).mean().dropna()

def RSI(data, n=14):
    d = data['Close']
    U = SMMA(np.maximum(data['Close'] - data['Close'].shift(1), 0).dropna(), n)
    D = SMMA(np.maximum(data['Close'].shift(1) - data['Close'], 0).dropna(), n)
    return 100 * U / (U + D + 1e-25)

def ADX(data, n=14):
    UpMove = (data['High'] - data['High'].shift(1)).dropna()
    DownMove = (data['Low'].shift(1) - data['Low']).dropna()
    
    plus_DM = np.where((UpMove > DownMove) & (UpMove > 0), UpMove, 0)
    minus_DM = np.where((UpMove < DownMove) & (DownMove > 0), DownMove, 0)
    
    return pd.Series(100 * SMMA(
        np.abs(SMMA(plus_DM, n) - SMMA(minus_DM, n)) /
        np.abs(SMMA(plus_DM, n) + SMMA(minus_DM, n)) + 1e-25,
        n
    ).values, index=UpMove.index)

def draw_rsi(ticker_list, figsize=(10, 7), source='obb'):
    rsi_data = {}
    for ticker in ((ticker_list).split('\n')):
        if source == 'obb':
            data = get_daily_data_with_today(ticker, datetime.datetime.combine(datetime.date.today(), datetime.time()), enforce_today=True)
        elif source == 'tcs':
            data = load_tcs_data(ticker, now() - datetime.timedelta(days=365), enforce_today=True)
        rsi_data[ticker] = RSI(data)
    rsi_data = pd.DataFrame(rsi_data)
    rsi_data.index = [str(x)[-5:].replace('-', '/') for x in rsi_data.index.date]
    if source == 'obb':
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