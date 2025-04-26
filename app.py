import streamlit as st
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, AutoDateLocator
import backtrader as bt
import numpy as np

# --- Import local modules ---
from utils.data_fetcher import get_stock_data
from strategies.quantum_momentum import QuantumMomentumStrategy  # Import the strategy

# --- Streamlit App Configuration ---
st.set_page_config(layout="wide")
st.title("Quantum-Inspired Trading Bot Backtester")

# --- Sidebar Controls ---
st.sidebar.header("Backtest Configuration")
ticker = st.sidebar.text_input("Ticker Symbol", "AAPL")  # Default ticker
start_date_str = st.sidebar.date_input("Start Date", datetime(2022, 1, 1))
end_date_str = st.sidebar.date_input("End Date", datetime(2023, 1, 1))
start_cash = st.sidebar.number_input(
    "Initial Cash", value=10000.0, min_value=100.0, step=1000.0
)

# Strategy Selection (Add more strategies to the dictionary as you create them)
available_strategies = {
    "QuantumMomentum": QuantumMomentumStrategy
    # "AnotherStrategy": AnotherStrategyClass
}
strategy_name = st.sidebar.selectbox(
    "Select Strategy", list(available_strategies.keys())
)
SelectedStrategy = available_strategies[strategy_name]

# Strategy-specific parameters (Example for QuantumMomentum)
st.sidebar.subheader(f"{strategy_name} Parameters")
if strategy_name == "QuantumMomentum":
    sma_period = st.sidebar.slider("SMA Period", 5, 50, 10)
    prob_threshold = st.sidebar.slider("Probability Threshold", 0.5, 0.95, 0.6, 0.01)
    strategy_params = {
        "sma_period": sma_period,
        "prob_threshold": prob_threshold,
    }
else:
    strategy_params = {}

run_button = st.sidebar.button("Run Backtest")

# --- Main Panel ---
st.header("Backtest Results")

if run_button:
    if not ticker:
        st.error("Please enter a ticker symbol.")
    else:
        start_date = pd.to_datetime(start_date_str)
        end_date = pd.to_datetime(end_date_str)

        # 1. Fetch Data
        with st.spinner(f"Fetching data for {ticker}..."):
            data = get_stock_data(ticker, start_date, end_date)

        if data is not None and not data.empty:
            st.success(
                f"Data fetched successfully for {ticker} from {start_date_str} to {end_date_str}."
            )

            # Prepare data for Backtesting.py (needs OHLC columns with specific names)
            # Print the original column names for debugging

            # Ensure we have the right column formats - explicitly rename for clarity
            data_cols = {col.lower(): col for col in data.columns}

            # Create a properly formatted dataframe for backtesting
            bt_data = data.copy()

            # Make sure columns are properly named for Backtesting.py
            rename_map = {}
            if "open" in data_cols:
                rename_map[data_cols["open"]] = "Open"
            if "high" in data_cols:
                rename_map[data_cols["high"]] = "High"
            if "low" in data_cols:
                rename_map[data_cols["low"]] = "Low"
            if "close" in data_cols:
                rename_map[data_cols["close"]] = "Close"
            if "volume" in data_cols:
                rename_map[data_cols["volume"]] = "Volume"
            if "adj_close" in data_cols or "adj close" in data_cols:
                key = data_cols.get("adj_close", data_cols.get("adj close"))
                rename_map[key] = "Adj Close"

            bt_data = bt_data.rename(columns=rename_map)

            # 2. Initialize and run backtrader Cerebro
            # Create a subclass with the strategy parameters
            if strategy_params:
                StrategyClass = type(
                    f"{strategy_name}_Custom", (SelectedStrategy,), strategy_params
                )
            else:
                StrategyClass = SelectedStrategy

            cerebro = bt.Cerebro()
            cerebro.addstrategy(StrategyClass, **strategy_params)
            # Convert pandas DataFrame to backtrader data feed
            data_feed = bt.feeds.PandasData(dataname=bt_data)
            cerebro.adddata(data_feed)
            cerebro.broker.setcash(start_cash)
            cerebro.broker.setcommission(commission=0.001)
            st.write("Running backtest with backtrader...")
            with st.spinner("Backtest in progress..."):
                strats = cerebro.run()
            st.write("Backtest complete.")
            # Extract strategy instance to access buy/sell signals
            strat = strats[0]

            # Get final portfolio value
            final_value = cerebro.broker.getvalue()
            results = {"Equity Final [$]": final_value}

            # 4. Display Basic Results
            st.subheader("Performance Metrics")
            # Use markdown for larger text and better layout control
            st.markdown(
                f"""
            <div style="display: flex; justify-content: space-around; font-size: 1.5em; margin-bottom: 20px;">
                <div style="text-align: center;">
                    <strong>Initial Portfolio Value</strong><br>
                    ${start_cash:,.2f}
                </div>
                <div style="text-align: center;">
                    <strong>Final Portfolio Value</strong><br>
                    ${final_value:,.2f}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # 5. Plot Price with Buy/Sell Signals
            st.subheader("Price Chart with Buy/Sell Signals")
            # Plot using Matplotlib
            # Apply professional built-in matplotlib style
            # plt.style.use("ggplot")
            # Set transparent background
            fig, ax = plt.subplots(figsize=(12, 6), facecolor="none")
            ax.set_facecolor("none")
            ax.grid(False)  # ← turn off all grid lines

            # Plot closing price with styling
            # Draw close price line with low z-order to sit behind markers
            ax.plot(
                bt_data.index,
                bt_data["Close"],
                label="Close Price",
                color="steelblue",
                linewidth=2,
                zorder=1,
            )
            # Add grid for readability
            # Improve date formatting
            locator = AutoDateLocator()
            formatter = DateFormatter("%Y-%m")
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            fig.autofmt_xdate()

            # Plot buy signals
            if hasattr(strat, "buy_signals") and strat.buy_signals:
                buy_dates, buy_prices = zip(*strat.buy_signals)
                # Black triangle for buy (drawn above line)
                ax.scatter(
                    buy_dates,
                    buy_prices,
                    marker="^",
                    color="white",
                    s=70,
                    label="Buys",
                    zorder=3,
                )

            # Plot sell signals
            if hasattr(strat, "sell_signals") and strat.sell_signals:
                sell_dates, sell_prices = zip(*strat.sell_signals)
                # Black triangle down for sell (drawn above line)
                ax.scatter(
                    sell_dates,
                    sell_prices,
                    marker="v",
                    color="grey",
                    s=70,
                    label="Sells",
                    zorder=3,
                )

            # Annotate profit/loss for each completed trade
            if hasattr(strat, "trades") and strat.trades:
                for trade in strat.trades:
                    dt = trade["sell_date"]
                    price = trade["sell_price"]
                    pnl = trade["pnl"]
                    text_color = "g" if pnl >= 0 else "r"
                    # Place P&L text above markers with highest z-order
                    ax.annotate(
                        f"{pnl:.2f}",
                        xy=(dt, price),
                        xytext=(0, 10),
                        textcoords="offset points",
                        ha="center",
                        color=text_color,
                        fontsize=12,
                        fontweight="bold",  # ← make annotation text bold
                        zorder=4,
                    )
                    # make spines white and bold
            for spine in ax.spines.values():
                spine.set_color("white")
                spine.set_linewidth(2)

            # ticks white and bold
            ax.tick_params(axis="x", colors="white", labelsize=12, width=2)
            ax.tick_params(axis="y", colors="white", labelsize=12, width=2)

            # axis labels bold and white
            ax.set_xlabel("Date", fontsize=12, color="white", weight="bold")
            ax.set_ylabel("Price", fontsize=12, color="white", weight="bold")
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Price", fontsize=12)
            ax.legend(loc="upper right", framealpha=0.7)
            # Ensure plot background is transparent when saving/displaying
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
            st.pyplot(fig)

        elif data is None:
            st.error(
                f"Could not fetch data for {ticker}. Please check the symbol and date range."
            )
        else:  # Data is empty
            st.warning(f"No data available for {ticker} in the specified date range.")

# --- How to Run ---
st.sidebar.markdown("---")
st.sidebar.info("To run: `streamlit run app.py` in your terminal.")
st.sidebar.info("Make sure to install backtrader package: `pip install backtrader`")
