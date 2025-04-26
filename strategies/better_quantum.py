import backtrader as bt
import numpy as np
from scipy import fft
from scipy.stats import norm


class CustomQuantumStrategy(bt.Strategy):
    """
    A simplified template for quantum-inspired trading strategies.
    Focuses on core quantum concepts with minimal parameters.
    """

    # Simplified parameter set
    params = (
        # Core parameters
        ("sma_period", 10),  # Period for moving average calculation
        ("prob_threshold", 0.6),  # Decision threshold
        # Essential quantum parameters
        ("quantum_factor", 0.5),  # Controls overall quantum effect strength
        ("phase_period", 20),  # Period for phase analysis
        ("uncertainty", 0.2),  # Uncertainty principle factor
        # Wavefunction parameters
        ("wf_components", 5),  # Number of wavefunction components to fit
        ("wf_lookback", 50),  # Lookback period for wavefunction fitting
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Trade tracking
        self.buy_signals = []
        self.sell_signals = []
        self.trades = []
        self.current_trade = None

        # Moving average indicator
        self.sma = bt.indicators.SMA(self.data.close, period=self.p.sma_period)

        # Quantum state (simplified)
        self.quantum_state = np.array([0.5, 0.5])  # Initial 50/50 state

        # Wavefunction storage
        self.wavefunction_data = []
        self.price_data = []
        self.dates = []

    def next(self):
        # Skip if not enough data
        if len(self.data) < self.p.sma_period:
            return

        # 1. MARKET STATE CALCULATION
        price = self.data.close[0]
        dt = self.data.datetime.datetime(0)

        # Store date and price for wavefunction visualization
        self.dates.append(dt)
        self.price_data.append(price)

        # Calculate z-score (normalized distance from moving average)
        closes = [self.data.close[-i] for i in range(self.p.sma_period)]
        sma = np.mean(closes)
        vol = np.std(closes) if np.std(closes) > 0 else 1
        z_score = (price - sma) / vol

        # 2. QUANTUM ANALYSIS
        # Calculate market phase (0-1 representing cycle position)
        phase = self._calculate_phase()

        # Calculate quantum amplitude (-1 to 1 representing market direction)
        amplitude = np.tanh(z_score) + self.p.quantum_factor * (
            2 * phase - 1
        ) * np.sign(z_score)
        amplitude = np.clip(amplitude, -1, 1)

        # Calculate buy/sell probabilities with uncertainty principle
        buy_prob = 0.5 * (amplitude + 1)
        sell_prob = 1.0 - buy_prob

        # Apply uncertainty (stronger signals have more timing uncertainty)
        certainty = abs(buy_prob - 0.5) * 2
        adjustment = self.p.uncertainty * certainty

        if buy_prob > 0.5:
            buy_prob -= adjustment
        else:
            buy_prob += adjustment

        # Fit wavefunction to current price data
        if len(self.data) >= self.p.wf_lookback:
            wf_value = self._fit_wavefunction()
            self.wavefunction_data.append(wf_value)
        else:
            self.wavefunction_data.append(
                price
            )  # Use price as placeholder when not enough data

        # 3. DECISION MAKING
        # Make trading decision
        if not self.position:  # Not in market
            if buy_prob > self.p.prob_threshold:
                # Buy with all available cash
                cash = self.broker.getcash() * 0.995
                shares = int(cash / price)
                if shares > 0:
                    self.buy(size=shares)
                    self.buy_signals.append((dt, price))
                    self.current_trade = {
                        "buy_date": dt,
                        "buy_price": price,
                        "shares": shares,
                    }
        else:  # In market
            if (1 - buy_prob) > self.p.prob_threshold:
                # Sell all shares
                self.close()
                self.sell_signals.append((dt, price))

                # Record trade
                if self.current_trade:
                    pnl = price - self.current_trade["buy_price"]
                    total_pnl = pnl * self.current_trade["shares"]
                    self.trades.append(
                        {
                            "buy_date": self.current_trade["buy_date"],
                            "buy_price": self.current_trade["buy_price"],
                            "sell_date": dt,
                            "sell_price": price,
                            "shares": self.current_trade["shares"],
                            "total_pnl": total_pnl,
                        }
                    )
                    self.current_trade = None

    def _calculate_phase(self):
        """Calculate market phase using simplified quantum phase estimation"""
        # Get price history
        lookback = self.p.phase_period * 2
        if len(self.data) < lookback:
            return 0.5  # Default to middle of cycle

        price_history = np.array([self.data.close[-i] for i in range(lookback)])

        # Simple Fourier transform to find dominant cycle
        fft_result = fft.fft(price_history)
        power = np.abs(fft_result) ** 2
        dominant_freq = np.argmax(power[1 : len(power) // 2]) + 1
        phase = np.angle(fft_result[dominant_freq])

        # Normalize to [0,1]
        return (phase + np.pi) / (2 * np.pi)

    def _fit_wavefunction(self):
        """Fit a quantum-inspired wavefunction to recent price history"""
        # Get price history for wavefunction fitting
        lookback = min(len(self.data), self.p.wf_lookback)
        if lookback < 10:  # Need minimum data points
            return self.data.close[0]

        price_history = np.array([self.data.close[-i] for i in range(lookback)])
        # Normalize price history
        mean_price = np.mean(price_history)
        std_price = np.std(price_history) if np.std(price_history) > 0 else 1
        normalized_price = (price_history - mean_price) / std_price

        # Apply FFT to get frequency components
        fft_result = fft.fft(normalized_price)

        # Get top frequency components (by amplitude)
        amplitudes = np.abs(fft_result[: lookback // 2])
        phases = np.angle(fft_result[: lookback // 2])

        # Sort by amplitude and get top components
        top_indices = np.argsort(amplitudes)[-self.p.wf_components :]

        # Reconstruct wavefunction using top components
        t = np.arange(lookback)
        wavefunction = np.zeros(lookback)

        for idx in top_indices:
            if idx > 0:  # Skip DC component
                freq = idx / lookback
                amplitude = amplitudes[idx] / lookback
                phase = phases[idx]
                wavefunction += amplitude * np.cos(2 * np.pi * freq * t + phase)

        # Scale back to price range and add mean
        wavefunction = (wavefunction * std_price) + mean_price

        # Return current fitted value (can be used for prediction)
        return wavefunction[0]

    def get_wavefunction_data(self):
        """Return the fitted wavefunction data along with dates and prices"""
        return {
            "dates": self.dates,
            "prices": self.price_data,
            "wavefunction": self.wavefunction_data,
        }
