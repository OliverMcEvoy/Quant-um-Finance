import backtrader as bt
import numpy as np


class QuantumMomentumStrategy(bt.Strategy):
    # Define parameters with default values
    sma_period = 10
    prob_threshold = 0.6

    def next(self):
        # Get current close price (latest bar)
        price = self.data.close[0]
        # Ensure enough data points for SMA
        if len(self.data) < self.sma_period:
            return
        # Calculate simple moving average for last sma_period bars, including current
        closes = [self.data.close[-i] for i in range(self.sma_period)]
        sma = np.mean(closes)
        price_above_sma = price > sma

        # Simplified 'probability' based on price vs SMA
        buy_probability = 0.75 if price_above_sma else 0.25
        sell_probability = 1.0 - buy_probability

        # Random draw for quantum-inspired decisions
        random_draw = np.random.rand()

        # Trading logic
        if not self.position:
            if buy_probability > self.prob_threshold and random_draw < buy_probability:
                self.buy()
        else:
            if (
                sell_probability > self.prob_threshold
                and random_draw < sell_probability
            ):
                self.sell()
