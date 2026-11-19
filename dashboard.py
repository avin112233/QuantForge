"""QuantForge research dashboard. Launch: streamlit run dashboard.py"""
from pathlib import Path
import pandas as pd
import streamlit as st
from src.dashboard_data import ROOT, read_json, read_csv, metric_table, ml_table, date_indexed, numeric_columns

st.set_page_config(page_title='QuantForge | Quant Research', page_icon='📈', layout='wide')
st.title('QuantForge')
st.caption('ETF research · Walk-forward ML · Portfolio optimization · Risk analytics')

root = ROOT / 'outputs'
portfolio = read_json(root / 'day5' / 'summary.json')
ml = read_json(root / 'day4' / 'summary.json')
metrics = metric_table(portfolio)
ml_metrics = ml_table(ml)
equity = numeric_columns(date_indexed(read_csv(root / 'day5' / 'equity_curves.csv')))
allocations = date_indexed(read_csv(root / 'day5' / 'allocations.csv'))

page = st.sidebar.radio('Research module', ['Overview', 'Portfolio Analytics', 'Asset Allocation', 'ML Research', 'Strategy Comparison', 'Reports'])
st.sidebar.caption('Research only. Backtests are not investment advice.')

if page == 'Overview':
    st.subheader('Portfolio research overview')
    if metrics.empty:
        st.info('Day 5 summary not found. Run: python day5.py')
    else:
        st.dataframe(metrics, use_container_width=True, hide_index=True)
        if 'total_return_pct' in metrics:
            best = metrics.loc[pd.to_numeric(metrics['total_return_pct'], errors='coerce').idxmax()]
            st.metric('Highest simulated total return', f"{best['total_return_pct']}%", best['strategy'])
    if not equity.empty:
        st.subheader('Portfolio equity curves')
        st.line_chart(equity)
    else:
        st.caption('No equity_curves.csv found in outputs/day5.')

elif page == 'Portfolio Analytics':
    st.subheader('Out-of-sample portfolio analytics')
    if not metrics.empty:
        available = [c for c in ['total_return_pct','annualized_return_pct','annualized_volatility_pct','sharpe','max_drawdown_pct','daily_var_95_pct','daily_cvar_95_pct'] if c in metrics.columns]
        if available:
            chosen = st.selectbox('Compare metric', available)
            chart = metrics.set_index('strategy')[[chosen]].apply(pd.to_numeric, errors='coerce')
            st.bar_chart(chart)
        st.dataframe(metrics, use_container_width=True, hide_index=True)
    else:
        st.info('Run Day 5 to generate summary.json.')
    if not equity.empty:
        st.line_chart(equity)

elif page == 'Asset Allocation':
    st.subheader('Portfolio allocations')
    if allocations.empty:
        st.info('No allocations.csv found. Run Day 5 first.')
    else:
        st.dataframe(allocations.tail(30), use_container_width=True)
        st.download_button('Download allocations CSV', allocations.to_csv(index=True).encode(), 'allocations.csv', 'text/csv')
        numeric = numeric_columns(allocations)
        if not numeric.empty:
            st.caption('Numeric allocation columns (format depends on Day 5 output)')
            st.line_chart(numeric)

elif page == 'ML Research':
    st.subheader('XGBoost versus buy-and-hold')
    if ml_metrics.empty:
        st.info('Day 4 summary not found. Run: python day4.py')
    else:
        st.dataframe(ml_metrics, use_container_width=True, hide_index=True)
        if {'ticker','ml_return_pct','hold_return_pct'}.issubset(ml_metrics.columns):
            st.bar_chart(ml_metrics.set_index('ticker')[['ml_return_pct','hold_return_pct']])
        st.caption('AUC near 0.5 suggests limited directional discrimination. Historical simulations do not guarantee future results.')

elif page == 'Strategy Comparison':
    st.subheader('Trading strategy research')
    st.caption('Inspect available Day 3 and Day 2 research artifacts.')
    for day in ('day3', 'day2'):
        folder = root / day
        st.markdown(f'**{day.upper()}**')
        csvs = sorted(folder.glob('*.csv')) if folder.is_dir() else []
        if csvs:
            selected = st.selectbox(f'{day} CSV', csvs, format_func=lambda p: p.name, key=day)
            frame = read_csv(selected)
            st.dataframe(frame.head(100), use_container_width=True)
        else:
            st.info(f'No CSV summaries found in outputs/{day}.')
        pngs = sorted(folder.glob('*.png')) if folder.is_dir() else []
        if pngs:
            selected_png = st.selectbox(f'{day} chart', pngs, format_func=lambda p: p.name, key=f'{day}_png')
            st.image(str(selected_png), use_container_width=True)

else:
    st.subheader('Download research reports')
    for day in ('day4', 'day5'):
        path = root / day / 'summary.json'
        if path.is_file():
            st.download_button(f'Download {day} summary JSON', path.read_bytes(), f'{day}_summary.json', 'application/json')
    if not metrics.empty:
        st.download_button('Download portfolio metrics CSV', metrics.to_csv(index=False).encode(), 'portfolio_metrics.csv', 'text/csv')
    if not ml_metrics.empty:
        st.download_button('Download ML metrics CSV', ml_metrics.to_csv(index=False).encode(), 'ml_metrics.csv', 'text/csv')
    st.caption('All metrics are historical/simulated and subject to data quality, execution assumptions, and model risk.')
