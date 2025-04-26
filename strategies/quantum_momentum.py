import backtrader as bt
import numpy as np


class QuantumMomentumStrategy(bt.Strategy):
    # Define strategy parameters using backtrader params
    params = (("sma_period", 10), ("prob_threshold", 0.6))

    def __init__(self, *args, **kwargs):
        # Initialize base strategy with params
        super().__init__(*args, **kwargs)
        # Lists to record buy and sell signals (datetime, price)
        self.buy_signals = []
        self.sell_signals = []
        # Track individual trades and current open trade
        self.trades = []  # Each trade: dict with buy/sell info and pnl
        self.current_trade = None

    def next(self):
        # Get current close price (latest bar)
        price = self.data.close[0]
        # Ensure enough data points for SMA
        if len(self.data) < self.p.sma_period:
            return
        # Calculate simple moving average for last sma_period bars, including current
        closes = [self.data.close[-i] for i in range(self.p.sma_period)]
        sma = np.mean(closes)
        # Quantum-inspired probability using z-score relative to volatility
        vol = np.std(closes) if np.std(closes) > 0 else 1
        z_score = (price - sma) / vol
        amplitude = np.tanh(z_score)
        buy_probability = 0.5 * (amplitude + 1)
        sell_probability = 1.0 - buy_probability

        # Random draw for quantum-inspired decisions
        random_draw = np.random.rand()

        # Trading logic
        if not self.position:
            if (
                buy_probability > self.p.prob_threshold
                and random_draw < buy_probability
            ):
                dt = self.data.datetime.datetime(0)
                self.buy()
                # Record buy signal and start a new trade
                self.buy_signals.append((dt, price))
                self.current_trade = {"buy_date": dt, "buy_price": price}
        else:
            if (
                sell_probability > self.p.prob_threshold
                and random_draw < sell_probability
            ):
                dt = self.data.datetime.datetime(0)
                self.sell()
                # Record sell signal
                self.sell_signals.append((dt, price))
                # Complete the trade and compute P&L
                if self.current_trade is not None:
                    buy_info = self.current_trade
                    pnl = price - buy_info["buy_price"]
                    trade = {
                        "buy_date": buy_info["buy_date"],
                        "buy_price": buy_info["buy_price"],
                        "sell_date": dt,
                        "sell_price": price,
                        "pnl": pnl,
                    }
                    self.trades.append(trade)
                self.current_trade = None

        # Note: backtrader datetime.datetime(0) returns current bar datetime; the code appends tuples for plotting
