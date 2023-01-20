import investpack
import streamlit as st

hk_tickers = """HKG:0388
HKG:3988
HKG:2331
HKG:0669
HKG:9988
HKG:2015
HKG:0857
HKG:0386
HKG:1088
HKG:1347
HKG:9999
HKG:1928
HKG:1109
HKG:0939
HKG:9866
HKG:9618
HKG:2688
HKG:0992
HKG:0268
HKG:2518
HKG:9888
HKG:9626
HKG:0700
HKG:1113"""

hk8_tickers = """HKG:0001
HKG:0175
HKG:2020
HKG:2269
HKG:2628
HKG:0288
HKG:0291
HKG:3888"""

b = investpack.draw_rsi(tickers)
st.image(
    [investpack.draw_rsi(hk_tickers), investpack.draw_rsi(hk8_tickers)],
    caption=['Main HKG stocks', 'BX8 HKG stocks']
)
