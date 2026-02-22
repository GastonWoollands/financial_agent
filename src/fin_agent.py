import json
from typing import Dict, Tuple, List, Optional
import yfinance as yf
import pandas as pd
from scipy.stats import spearmanr
from agno.tools import Toolkit
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datetime import datetime
import numpy as np
from scipy.stats import norm

class YFinanceTools(Toolkit):
    """A toolkit for retrieving financial data using the yFinance API."""

    def __init__(
        self,
        stock_price: bool = True,
        company_info: bool = False,
        stock_fundamentals: bool = False,
        income_statements: bool = False,
        key_financial_ratios: bool = False,
        analyst_recommendations: bool = False,
        company_news: bool = False,
        technical_indicators: bool = False,
        historical_prices: bool = False,
        correlation: bool = False,
        volatility: bool = True,
        options_sentiment: bool = False,
        historical_evolution: bool = False,
        black_scholes_pricing: bool = False,
        options_chain_with_metrics: bool = False,
        enable_all: bool = False,
        max_workers: int = 5,
        request_timeout: int = 10,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        super().__init__(name="yfinance_tools")
        self.max_workers = max_workers
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        if stock_price or enable_all:
            self.register(self.get_current_stock_price)
        if company_info or enable_all:
            self.register(self.get_company_info)
        if stock_fundamentals or enable_all:
            self.register(self.get_stock_fundamentals)
        if income_statements or enable_all:
            self.register(self.get_income_statements)
        if key_financial_ratios or enable_all:
            self.register(self.get_key_financial_ratios)
        if analyst_recommendations or enable_all:
            self.register(self.get_analyst_recommendations)
        if company_news or enable_all:
            self.register(self.get_company_news)
        if technical_indicators or enable_all:
            self.register(self.get_technical_indicators)
        if historical_prices or enable_all:
            self.register(self.get_historical_stock_prices)
        if correlation or enable_all:
            self.register(self.get_correlation)
        if volatility or enable_all:
            self.register(self.get_volatility)
        if options_sentiment or enable_all:
            self.register(self.get_options_sentiment)
        if historical_evolution or enable_all:
            self.register(self.get_historical_comparison)
        if black_scholes_pricing or enable_all:
            self.register(self.get_black_scholes_pricing)
        if options_chain_with_metrics or enable_all:
            self.register(self.get_options_chain_with_metrics)

    def _retry_with_backoff(self, func, *args, **kwargs):
        """Implement exponential backoff for retrying failed requests."""
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                time.sleep(self.retry_delay * (2 ** attempt))

    @lru_cache(maxsize=100)
    def _fetch_ticker(self, symbol: str) -> yf.Ticker:
        """Fetch a yf.Ticker object with error handling and caching."""
        try:
            return self._retry_with_backoff(yf.Ticker, symbol)
        except Exception as e:
            raise ValueError(f"Failed to fetch ticker for {symbol}: {str(e)}")

    def _fetch_multiple_tickers(self, symbols: List[str]) -> Dict[str, yf.Ticker]:
        """Fetch multiple tickers in parallel."""
        tickers = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_symbol = {
                executor.submit(self._fetch_ticker, symbol): symbol 
                for symbol in symbols
            }
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    tickers[symbol] = future.result()
                except Exception as e:
                    tickers[symbol] = None
        return tickers

    def _to_json(self, data, error_msg: str = None) -> str:
        """Convert data to JSON string or return an error message."""
        if data is not None and (isinstance(data, dict) or isinstance(data, list) or not pd.isna(data)):
            return json.dumps(data, indent=2, default=str)
        return json.dumps({"error": error_msg}, indent=2)

    def _compute_returns(self, data: pd.Series) -> pd.Series:
        """Compute percentage change returns, dropping NaN values."""
        return data.pct_change().dropna()

    def _get_annualized_volatility(self, symbol: str, period: str = "1y", interval: str = "1d") -> Optional[float]:
        """Compute annualized volatility (HV) for a symbol. Returns float e.g. 0.25 for 25%, or None on failure."""
        try:
            ticker = self._fetch_ticker(symbol)
            data = self._retry_with_backoff(
                lambda: ticker.history(period=period, interval=interval)
            )
            if data.empty or len(data) < 2:
                return None
            returns = self._compute_returns(data["Close"])
            if returns.empty:
                return None
            volatility = returns.std() * (252 ** 0.5)
            return float(volatility) if not pd.isna(volatility) and volatility > 0 else None
        except Exception:
            return None

    def _vectorized_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators using vectorized operations."""
        # Moving Averages
        df["sma_20"] = df["close"].rolling(20, min_periods=1).mean()
        df["sma_50"] = df["close"].rolling(50, min_periods=1).mean()
        df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()

        # RSI
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
        rs = gain / loss.replace(0, 1e-10)
        df["rsi"] = 100 - (100 / (1 + rs))

        # MACD
        df["macd"] = df["close"].ewm(span=12, adjust=False).mean() - df["close"].ewm(span=26, adjust=False).mean()
        df["signal_line"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["signal_line"]

        # Bollinger Bands
        df["bb_middle"] = df["close"].rolling(20, min_periods=1).mean()
        bb_std = df["close"].rolling(20, min_periods=1).std()
        df["bb_upper"] = df["bb_middle"] + 2 * bb_std
        df["bb_lower"] = df["bb_middle"] - 2 * bb_std

        # VWAP
        tp = (df["high"] + df["low"] + df["close"]) / 3
        df["vwap"] = (tp * df["volume"]).cumsum() / df["volume"].cumsum()

        # ATR
        tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14, min_periods=1).mean()

        # Stochastic Oscillator
        low_14, high_14 = df["low"].rolling(14, min_periods=1).min(), df["high"].rolling(14, min_periods=1).max()
        df["stoch_k"] = 100 * (df["close"] - low_14) / (high_14 - low_14).replace(0, 1e-10)
        df["stoch_d"] = df["stoch_k"].rolling(3, min_periods=1).mean()

        # ADX
        plus_dm = (df["high"] - df["high"].shift()).clip(lower=0)
        minus_dm = (df["low"].shift() - df["low"]).clip(lower=0)
        tr_smooth = tr.rolling(14, min_periods=1).mean()
        plus_di = 100 * plus_dm.rolling(14, min_periods=1).mean() / tr_smooth
        minus_di = 100 * minus_dm.rolling(14, min_periods=1).mean() / tr_smooth
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10)
        df["adx"] = dx.rolling(14, min_periods=1).mean()

        # OBV
        df["obv"] = (df["volume"] * (df["close"].gt(df["close"].shift()).astype(int) - 
                                    df["close"].lt(df["close"].shift()).astype(int))).cumsum()

        return df

    def get_current_stock_price(self, symbol: str) -> str:
        """Get current stock price with retry mechanism."""
        try:
            ticker = self._fetch_ticker(symbol)
            price = self._retry_with_backoff(
                lambda: ticker.info.get("regularMarketPrice") or ticker.info.get("currentPrice")
            )
            return self._to_json({"price": round(price, 4)}, f"No price data available for {symbol}")
        except Exception as e:
            return self._to_json(None, f"Error fetching price for {symbol}: {str(e)}")

    def get_company_info(self, symbol: str) -> str:
        """Get company information with parallel processing for multiple symbols."""
        try:
            ticker = self._fetch_ticker(symbol)
            info = self._retry_with_backoff(lambda: ticker.info)
            company_data = {
                k: info.get(v) for k, v in {
                    "name": "shortName",
                    "symbol": "symbol",
                    "price": "regularMarketPrice",
                    "market_cap": "marketCap",
                    "sector": "sector",
                    "industry": "industry",
                    "website": "website",
                    "summary": "longBusinessSummary"
                }.items()
            }
            company_data["price"] = f"{company_data['price'] or info.get('currentPrice', 'N/A')} {info.get('currency', 'USD')}"
            return self._to_json(company_data, f"No company info available for {symbol}")
        except Exception as e:
            return self._to_json(None, f"Error fetching company info for {symbol}: {str(e)}")

    def get_stock_fundamentals(self, symbol: str) -> str:
        """Use this function to get fundamental data for a given stock symbol yfinance API.

        Args:
            symbol (str): The stock symbol.
        Returns:
            dict: JSON containing fundamental data.
        """
        try:
            ticker = self._fetch_ticker(symbol)
            info = ticker.info
            fundamentals = {
                k: info.get(v, "N/A") for k, v in {
                    "symbol": "symbol",
                    "name": "longName",
                    "sector": "sector",
                    "industry": "industry",
                    "market_cap": "marketCap",
                    "pe_ratio": "forwardPE",
                    "pb_ratio": "priceToBook",
                    "eps": "trailingEps",
                    "beta": "beta",
                    "price_to_sales": "priceToSalesTrailing12Months",
                    "enterprise_value": "enterpriseValue",
                    "enterprise_to_revenue": "enterpriseToRevenue",
                    "enterprise_to_ebitda": "enterpriseToEbitda",
                    "profit_margins": "profitMargins",
                    "gross_margins": "grossMargins",
                    "operating_margins": "operatingMargins",
                    "ebitda_margins": "ebitdaMargins",
                }.items()
            }
            return self._to_json(fundamentals, f"No fundamental data available for {symbol}")
        except Exception as e:
            return self._to_json(None, f"Error fetching fundamentals for {symbol}: {str(e)}")
    
    def get_income_statements(self, symbol: str) -> str:
        """Use this function to get income statements for a given stock symbol.

        Args:
            symbol (str): The stock symbol.

        Returns:
            dict: JSON containing income statements or an empty dictionary.
        """
        try:
            ticker = self._fetch_ticker(symbol)
            financials = ticker.financials
            if financials.empty:
                return self._to_json(None, f"No income statements available for {symbol}")
            return self._to_json(financials.to_dict(orient="index"))
        except Exception as e:
            return self._to_json(None, f"Error fetching income statements for {symbol}: {str(e)}")

    def get_key_financial_ratios(self, symbol: str) -> str:
        """Use this function to get key financial ratios for a given stock symbol.

        Args:
            symbol (str): The stock symbol.

        Returns:
            dict: JSON containing key financial ratios.
        """
        try:
            ticker = self._fetch_ticker(symbol)
            info = ticker.info
            ratios = {
                k: info.get(v, "N/A") for k, v in {
                    "pe_ratio": "trailingPE",
                    "forward_pe_ratio": "forwardPE",
                    "pb_ratio": "priceToBook",
                    "price_to_sales": "priceToSalesTrailing12Months",
                    "debt_to_equity": "debtToEquity",
                    "return_on_assets": "returnOnAssets",
                    "return_on_equity": "returnOnEquity",
                    "earnings_growth": "earningsGrowth",
                    "revenue_growth": "revenueGrowth",
                    "trailing_peg_ratio": "trailingPegRatio",
                    "quick_ratio": "quickRatio",
                    "current_ratio": "currentRatio",
                    "free_cashflow": "freeCashflow",
                    "operating_cashflow": "operatingCashflow",
                    "total_cash": "totalCash",
                    "total_cash_per_share": "totalCashPerShare"
                }.items()
            }
            return self._to_json(ratios, f"No financial ratios available for {symbol}")
        except Exception as e:
            return self._to_json(None, f"Error fetching key financial ratios for {symbol}: {str(e)}")

    def get_liquidity_ratios(self, symbol: str) -> str:
        """Use this function to get key liquidity ratios for a given stock symbol.

        Args:
            symbol (str): The stock symbol.

        Returns:
            dict: JSON containing key liquidity ratios.
        """
        try:
            ticker = self._fetch_ticker(symbol)
            info = ticker.info
            liquidity_ratios = {
                k: info.get(v, "N/A") for k, v in {
                    "quick_ratio": "quickRatio",
                    "current_ratio": "currentRatio"
                }.items()
            }
            return self._to_json(liquidity_ratios, f"No liquidity ratios available for {symbol}")
        except Exception as e:
            return self._to_json(None, f"Error fetching liquidity ratios for {symbol}: {str(e)}")

    def get_cashflow_ratios(self, symbol: str) -> str:
        """Use this function to get key cashflows ratios for a given stock symbol.

        Args:
            symbol (str): The stock symbol.

        Returns:
            dict: JSON containing key cashflows ratios.
        """
        try:
            ticker = self._fetch_ticker(symbol)
            info = ticker.info
            cashflow_ratios = {
                k: info.get(v, "N/A") for k, v in {
                    "free_cashflow": "freeCashflow",
                    "operating_cashflow": "operatingCashflow",
                    "total_cash": "totalCash",
                    "total_cash_per_share": "totalCashPerShare"
                }.items()
            }
            return self._to_json(cashflow_ratios, f"No cash flow data available for {symbol}")
        except Exception as e:
            return self._to_json(None, f"Error fetching cash flow ratios for {symbol}: {str(e)}")

    def get_analyst_recommendations(self, symbol: str) -> str:
        """Use this function to get analyst recommendations for a given stock symbol.

        Args:
            symbol (str): The stock symbol.

        Returns:
            str: JSON containing analyst recommendations.
        """
        try:
            ticker = self._fetch_ticker(symbol)
            recs = ticker.recommendations
            if recs.empty:
                return self._to_json(None, f"No analyst recommendations available for {symbol}")
            return self._to_json(recs.to_dict(orient="index"))
        except Exception as e:
            return self._to_json(None, f"Error fetching analyst recommendations for {symbol}: {str(e)}")

    def get_company_news(self, symbol: str, num_stories: int = 3) -> str:
        """Use this function to get company news and press releases for a given stock symbol.

        Args:
            symbol (str): The stock symbol.
            num_stories (int): The number of news stories to return. Defaults to 5.

        Returns:
            str: JSON containing company news and press releases.
        """
        try:
            ticker = self._fetch_ticker(symbol)
            news = ticker.news[:num_stories]
            return self._to_json(news, f"No news available for {symbol}")
        except Exception as e:
            return self._to_json(None, f"Error fetching company news for {symbol}: {str(e)}")

    def get_technical_indicators(self, symbol: str, period: str = "1y", interval: str = "1d") -> str:
        """Use this function to get technical indicators from a given stock symbol.
        Args:
            symbol (str): The stock symbol.
            period (str, Default: "1y"): The period for which to retrieve historical prices. Defaults to "1y".
            interval (str, Default: "1d"): The interval between data points. Defaults to "1d".

        Returns:
            str: JSON containing technical indicators.
        
        """
        try:
            ticker = self._fetch_ticker(symbol)
            data = self._retry_with_backoff(
                lambda: ticker.history(period=period, interval=interval)
            )
            if data.empty:
                return self._to_json(None, f"No data for {symbol}")

            df = data.rename(columns={"Close": "close", "High": "high", "Low": "low", "Volume": "volume"})
            df = self._vectorized_technical_indicators(df)

            # Select and format indicators
            indicators = df[["close", "sma_20", "sma_50", "ema_12", "rsi", "macd", "signal_line", "macd_histogram",
                            "bb_upper", "bb_middle", "bb_lower", "vwap", "atr", "stoch_k", "stoch_d", "adx", "obv", "volume"]]
            indicators.index = indicators.index.strftime("%Y-%m-%d")
            indicators = indicators.dropna()

            latest = indicators.iloc[-1].to_dict()
            signals = {
                "rsi": "sell" if latest["rsi"] > 70 else "buy" if latest["rsi"] < 30 else "hold",
                "macd": "buy" if latest["macd"] > latest["signal_line"] else "sell",
                "bollinger": "sell" if latest["close"] > latest["bb_upper"] else "buy" if latest["close"] < latest["bb_lower"] else "hold"
            }

            result = {
                "metadata": {
                    "symbol": symbol,
                    "period": period,
                    "interval": interval,
                    "data_points": len(indicators)
                },
                "indicators": indicators.to_dict(orient="index"),
                "last_signals": signals
            }
            return self._to_json(result)
        except Exception as e:
            return self._to_json(None, f"Error fetching indicators for {symbol}: {str(e)}")

    def get_historical_stock_prices(self, symbol: str, period: str = "1mo", interval: str = "1d") -> str:
        """
        Use this function to get the historical stock price for a given symbol.

        Args:
            symbol (str): The stock symbol.
            period (str, Default: "1mo"): The period for which to retrieve historical prices. Defaults to "1mo".
                        Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
            interval (str, Default: "1d"): The interval between data points. Defaults to "1d".
                        Valid intervals: 1d,5d,1wk,1mo,3mo

        Returns:
          str: The current stock price or error message.
        """
        try:
            ticker = self._fetch_ticker(symbol)
            history = ticker.history(period=period, interval=interval)
            if history.empty:
                return self._to_json(None, f"No historical prices available for {symbol}")
            return self._to_json(history.to_dict(orient="index"))
        except Exception as e:
            return self._to_json(None, f"Error fetching historical prices for {symbol}: {str(e)}")

    def get_correlation(self, symbol_1: str, symbol_2: str, period: str = "1y", interval: str = "1d") -> str:
        """Use this function to get correlation between two assets.
        
        Args:
            symbol_1 (str): The first stock symbol.
            symbol_2 (str): The second stock symbol.
            period (str, Default: "1y"): The period for which to retrieve historical prices. Defaults to "1y".
            interval (str, Default: "1d"): The interval between data points. Defaults to "1d".
        """
        try:
            tickers = self._fetch_multiple_tickers([symbol_1, symbol_2])
            if None in tickers.values():
                return self._to_json(None, f"Failed to fetch data for one or both symbols")

            data1 = self._retry_with_backoff(
                lambda: tickers[symbol_1].history(period=period, interval=interval)["Close"]
            )
            data2 = self._retry_with_backoff(
                lambda: tickers[symbol_2].history(period=period, interval=interval)["Close"]
            )

            if data1.empty or data2.empty:
                return self._to_json(None, f"No data available for {symbol_1} or {symbol_2}")

            returns1 = self._compute_returns(data1)
            returns2 = self._compute_returns(data2)
            aligned = returns1.align(returns2, join="inner")
            pearson = aligned[0].corr(aligned[1])
            spearman, _ = spearmanr(aligned[0], aligned[1])

            metadata = {
                "symbol_1": symbol_1,
                "symbol_2": symbol_2,
                "period": period,
                "interval": interval,
                "data_points": len(aligned[0]),
                "pearson_correlation": round(pearson, 4) if not pd.isna(pearson) else "N/A",
                "spearman_correlation": round(spearman, 4) if not pd.isna(spearman) else "N/A"
            }
            return self._to_json(metadata)
        except Exception as e:
            return self._to_json(None, f"Error calculating correlation for {symbol_1}-{symbol_2}: {str(e)}")

    def get_volatility(self, symbol: str, period: str = "1y", interval: str = "1d", benchmark_symbol: str = "^SPX") -> str:
        """Use this function to get volatility metrics.
        
        Args:
            symbol (str): The stock symbol.
            period (str, Default: "1y"): The period for which to retrieve historical prices. Defaults to "1y".
            interval (str, Default: "1d"): The interval between data points. Defaults to "1d".
            benchmark_symbol (str, Default: "^SPX"): The benchmark symbol. Defaults to "^SPX".
        """
        try:
            tickers = self._fetch_multiple_tickers([symbol, benchmark_symbol])
            if None in tickers.values():
                return self._to_json(None, f"Failed to fetch data for one or both symbols")

            data = self._retry_with_backoff(
                lambda: tickers[symbol].history(period=period, interval=interval)
            )
            bench_data = self._retry_with_backoff(
                lambda: tickers[benchmark_symbol].history(period=period, interval=interval)
            )

            if data.empty or bench_data.empty:
                return self._to_json(None, f"No data for {symbol} or {benchmark_symbol}")

            info = tickers[symbol].info
            current_price = info.get("regularMarketPrice", data["Close"].iloc[-1])
            target_price = info.get("targetMeanPrice", current_price)

            # Compute returns using vectorized operations
            returns = self._compute_returns(data["Close"])
            bench_returns = self._compute_returns(bench_data["Close"])
            aligned = returns.align(bench_returns, join="inner")
            asset_returns, bench_returns = aligned

            # Volatility metrics
            volatility = returns.std() * (252 ** 0.5)  # Annualized
            annualized_return = returns.mean() * 252
            sharpe = annualized_return / volatility if volatility != 0 else 0

            # Beta
            beta = asset_returns.cov(bench_returns) / bench_returns.var() if bench_returns.var() != 0 else 0

            # ATR
            tr = pd.DataFrame({
                "hl": data["High"] - data["Low"],
                "hc": (data["High"] - data["Close"].shift()).abs(),
                "lc": (data["Low"] - data["Close"].shift()).abs()
            }).max(axis=1)
            atr = tr.rolling(window=14, min_periods=1).mean().iloc[-1]

            # Max Drawdown
            cumulative_returns = (1 + returns).cumprod()
            peak = cumulative_returns.cummax()
            max_drawdown = ((cumulative_returns - peak) / peak).min()

            # Risk-Reward and Upside Potential
            upside_potential = ((target_price - current_price) / current_price) * 100 if current_price != 0 else 0
            risk_reward = abs(upside_potential / (max_drawdown * 100)) if max_drawdown != 0 else 0

            metadata = {
                "symbol": symbol,
                "benchmark": benchmark_symbol,
                "period": period,
                "interval": interval,
                "start_date": str(data.index[0].date()),
                "end_date": str(data.index[-1].date()),
                "data_points": len(returns),
                "volatility_annualized": f"{round(volatility * 100, 2)}%",
                "atr_latest": round(atr, 4),
                "sharpe_ratio": round(sharpe, 4),
                "max_drawdown": f"{round(max_drawdown * 100, 2)}%",
                "beta": round(beta, 2),
                "risk_reward_ratio": round(risk_reward, 2),
                "current_price": round(current_price, 2),
                "target_price": round(target_price, 2),
                "upside_potential_percent": round(upside_potential, 2)
            }
            return self._to_json(metadata)
        except Exception as e:
            return self._to_json(None, f"Error calculating volatility for {symbol}: {str(e)}")

    def get_options_sentiment(self, symbol: str, max_expirations: int = 5, iv_threshold: float = 0.3, skew_threshold: float = 0.1) -> str:
        """Use this function to fetch and analyze options chain data to determine bullish or bearish sentiment for a stock.
        
        Args:
            symbol (str): The stock ticker symbol (e.g., 'AAPL' for Apple).
            max_expirations (int, default=5): The maximum number of upcoming expiration dates to analyze. Default 5
            v_threshold (float, default=0.3): The implied volatility threshold used in sentiment calculations. Default 0.3
            skew_threshold (float, default=0.1): The minimum IV skew required to influence sentiment scoring. Default 0.1
        Returns:
            str: A JSON string with options metrics.
                 Returns an error message in JSON format if data retrieval fails.
        """
        try:
            ticker = self._fetch_ticker(symbol)
            expirations = ticker.options[:max_expirations]
            if not expirations:
                return self._to_json(None, f"No options data available for {symbol}")

            sentiment_data = {}
            for date in expirations:
                chain = ticker.option_chain(date)
                calls, puts = chain.calls, chain.puts

                call_oi = calls.get("openInterest", pd.Series([0])).sum()
                put_oi = puts.get("openInterest", pd.Series([0])).sum()
                pc_oi_ratio = put_oi / call_oi if call_oi > 0 else float("inf")

                call_vol = calls.get("volume", pd.Series([0])).sum()
                put_vol = puts.get("volume", pd.Series([0])).sum()
                pc_vol_ratio = put_vol / call_vol if call_vol > 0 else float("inf")

                call_iv = calls.get("impliedVolatility", pd.Series([0])).mean()
                put_iv = puts.get("impliedVolatility", pd.Series([0])).mean()
                avg_iv = (call_iv + put_iv) / 2 if call_iv and put_iv else max(call_iv, put_iv)
                iv_skew = put_iv - call_iv if call_iv and put_iv else 0

                score = sum([
                    -1 if pc_oi_ratio > 1.2 else 1 if pc_oi_ratio < 0.8 else 0,
                    -1 if pc_vol_ratio > 1.2 else 1 if pc_vol_ratio < 0.8 else 0,
                    -1 if avg_iv > iv_threshold and iv_skew > skew_threshold else 1 if avg_iv > iv_threshold else 0
                ])

                sentiment = {
                    "strong bullish": score >= 2,
                    "bullish": score == 1,
                    "strong bearish": score <= -2,
                    "bearish": score == -1
                }.get(True, "neutral")

                sentiment_data[date] = {
                    "put_call_oi_ratio": round(pc_oi_ratio, 2) if call_oi > 0 else "N/A",
                    "put_call_vol_ratio": round(pc_vol_ratio, 2) if call_vol > 0 else "N/A",
                    "avg_implied_volatility": round(avg_iv * 100, 2),
                    "iv_skew": round(iv_skew * 100, 2),
                    "sentiment_score": score,
                    "sentiment": sentiment
                }

            result = {
                "symbol": symbol,
                "expirations_analyzed": len(expirations),
                "sentiment": sentiment_data
            }
            return self._to_json(result)
        except Exception as e:
            return self._to_json(None, f"Error fetching options sentiment for {symbol}: {str(e)}")


    def get_historical_comparison(self, symbol: str, years: int = 5) -> str:
        """Use this function to compare stock's current P/E ratio, dividend yield, and revenue growth to their historical averages.

        Args:
            symbol (str): The stock ticker symbol (e.g., 'AAPL' for Apple).
            years (int: Default 5): Number of years for historical data (default is 5).

        Returns:
            str: A JSON string with current metrics, historical averages, and comparisons.
                 Returns an error message in JSON format if data retrieval fails.
        """
        try:
            stock = self._fetch_ticker(symbol)
            financials = stock.financials
            if financials.empty:
                return self._to_json(None, f"No financial data for {symbol}")

            # Filter columns with sufficient data and extract dates
            financials = financials.loc[:, financials.notna().sum() > 1].dropna(how="all")
            if len(financials.columns) < 2:
                return self._to_json(None, f"Insufficient historical data for {symbol}")

            dates = pd.to_datetime(financials.columns).tz_localize(None)

            eps_history = financials.loc["Basic EPS"] if "Basic EPS" in financials.index else None
            revenue_history = financials.loc["Total Revenue"] if "Total Revenue" in financials.index else None
            
            if eps_history is None or revenue_history is None:
                return self._to_json(None, f"Missing EPS or revenue data for {symbol}")

            # Fetch prices efficiently for fiscal year-ends
            hist = stock.history(start=min(dates) - pd.Timedelta(30, "D"), end=max(dates) + pd.Timedelta(1, "D"))
            hist.index = hist.index.tz_localize(None)

            if hist.empty:
                return self._to_json(None, f"No historical price data for {symbol}")

            prices = {date: hist["Close"].asof(date) for date in dates if hist["Close"].asof(date) is not pd.NaT}

            # P/E Ratio History
            pe_history = {date: prices[date] / eps if eps != 0 else None
                        for date, eps in eps_history.items() if date in prices and not pd.isna(eps)}
            valid_pe = [pe for pe in pe_history.values() if pe is not None]
            avg_pe = sum(valid_pe) / len(valid_pe) if valid_pe else None

            # Dividend Yield History
            dividends = stock.dividends

            if dividends.empty:
                avg_div_yield = None
            else:
                div_by_year = dividends.groupby(dividends.index.year).sum()
                div_yield_history = {
                    year: (div_by_year.get(year, 0) / hist["Close"].asof(f"{year}-12-31")) * 100
                    for year in range(min(dividends.index).year, max(dates).year)
                    if hist["Close"].asof(f"{year}-12-31") is not pd.NaT
                }
                avg_div_yield = sum(div_yield_history.values()) / len(div_yield_history) if div_yield_history else None

            # Revenue Growth History
            revenue_growth = ((revenue_history - revenue_history.shift(1)) / revenue_history.shift(1) * 100).dropna()
            avg_revenue_growth = revenue_growth.mean() if not revenue_growth.empty else None

            # Current Metrics
            info = stock.info
            current_pe = info.get("trailingPE")
            current_div_yield = info.get("dividendYield")
            if current_div_yield is not None:
                current_div_yield *= 100
            current_revenue_growth = info.get("revenueGrowth")
            if current_revenue_growth is not None:
                current_revenue_growth *= 100

            # Comparison
            comparison = {
                "symbol": symbol,
                "years_analyzed": years,
                "data_points": {
                    "pe_ratio": len(valid_pe) if valid_pe else 0,
                    "dividend_yield": len(div_yield_history) if avg_div_yield else 0,
                    "revenue_growth": len(revenue_growth),
                }
            }
            
            def compare(current, avg):
                return ("below average" if current < avg else "above average" if current > avg else "average")
        
            comparison = {
                "symbol": symbol,
                "years_analyzed": years,
                "data_points": {
                    "pe_ratio": len(pe_history),
                    "dividend_yield": len(div_yield_history) if avg_div_yield else 0,
                    "revenue_growth": len(revenue_growth),
                },
                "pe_ratio": {
                    "current": round(current_pe, 2) if current_pe else None,
                    "historical_average": round(avg_pe, 2) if avg_pe else None,
                    "comparison": compare(current_pe, avg_pe) if current_pe and avg_pe else None
                },
                "dividend_yield": {
                    "current": round(current_div_yield, 2) if current_div_yield else None,
                    "historical_average": round(avg_div_yield, 2) if avg_div_yield else None,
                    "comparison": compare(current_div_yield, avg_div_yield) if current_div_yield and avg_div_yield else None
                },
                "revenue_growth": {
                    "current": round(current_revenue_growth, 2) if current_revenue_growth else None,
                    "historical_average": round(avg_revenue_growth, 2) if avg_revenue_growth else None,
                    "comparison": compare(current_revenue_growth, avg_revenue_growth) if current_revenue_growth and avg_revenue_growth else None
                }
            }

            return self._to_json(comparison)
        except Exception as e:
            return self._to_json(None, f"Error processing {symbol}: {str(e)}")

    def get_black_scholes_pricing(self, symbol: str, strike: float, expiration_date: str, option_type: str = "call") -> str:
            """Calculates the theoretical Black-Scholes price for an option.
            
            Args:
                symbol (str): Stock ticker.
                strike (float): Strike price of the option.
                expiration_date (str): Expiration date in 'YYYY-MM-DD' format.
                option_type (str): "call" or "put".
            """
            try:
                ticker = self._fetch_ticker(symbol)
                
                # 1. Get Current Underlying Price (S)
                S = ticker.info.get("regularMarketPrice") or ticker.info.get("currentPrice")
                
                # 2. Get Risk-Free Rate (r) - Using 4% as a standard placeholder or fetch ^TNX
                r = 0.04 
                
                # 3. Calculate Time to Maturity (T) in years
                exp_date = datetime.strptime(expiration_date, "%Y-%m-%d")
                T = (exp_date - datetime.now()).days / 365.0
                if T <= 0:
                    return self._to_json(None, "Expiration date must be in the future.")

                # 4. Get Implied Volatility (sigma) from the option chain
                chain = ticker.option_chain(expiration_date)
                options_df = chain.calls if option_type.lower() == "call" else chain.puts
                
                # Find the closest strike to get a representative IV
                idx = (options_df['strike'] - strike).abs().idxmin()
                sigma = options_df.loc[idx, 'impliedVolatility']
                last_price = options_df.loc[idx, 'lastPrice']

                # 5. Black-Scholes Calculation
                d1 = (np.log(S / strike) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                d2 = d1 - sigma * np.sqrt(T)

                if option_type.lower() == "call":
                    price = (S * norm.cdf(d1)) - (strike * np.exp(-r * T) * norm.cdf(d2))
                    delta = norm.cdf(d1)
                else:
                    price = (strike * np.exp(-r * T) * norm.cdf(-d2)) - (S * norm.cdf(-d1))
                    delta = norm.cdf(d1) - 1

                result = {
                    "symbol": symbol,
                    "option_type": option_type,
                    "last_price": round(last_price, 4),
                    "theoretical_price": round(price, 4),
                    "delta": round(delta, 4),
                    "market_iv": round(sigma, 4),
                    "params": {
                        "spot_price": S,
                        "strike": strike,
                        "years_to_expiry": round(T, 4),
                        "risk_free_rate": r
                    }
                }
                hv = self._get_annualized_volatility(symbol)
                if hv is not None and hv > 0:
                    result["hv_pct"] = round(hv * 100, 2)
                    iv_hv_ratio = sigma / hv
                    result["iv_hv_ratio"] = round(iv_hv_ratio, 2)
                    result["rich_cheap"] = "Rich" if iv_hv_ratio > 1.1 else ("Cheap" if iv_hv_ratio < 0.9 else "Fair")
                return self._to_json(result)

            except Exception as e:
                return self._to_json(None, f"Error calculating Black-Scholes: {str(e)}")

    def get_options_chain_with_metrics(
        self,
        symbol: str,
        expiration_date: str,
        option_type: str = "call",
        max_strikes: int = 10,
        period: str = "1y",
    ) -> str:
        """Returns a list of options for an expiration with Strike, Last, BS price, IV%, HV%, IV/HV, Rich/Fair/Cheap, Delta.

        Args:
            symbol (str): Stock ticker.
            expiration_date (str): Expiration date in 'YYYY-MM-DD' format.
            option_type (str): "call" or "put".
            max_strikes (int): Maximum number of strikes to return (default 10).
            period (str): Period for historical volatility (default "1y").
        """
        try:
            ticker = self._fetch_ticker(symbol)
            exp_date = datetime.strptime(expiration_date, "%Y-%m-%d")
            if (exp_date - datetime.now()).days <= 0:
                return self._to_json(None, "Expiration date must be in the future.")

            S = ticker.info.get("regularMarketPrice") or ticker.info.get("currentPrice")
            r = 0.04
            T = (exp_date - datetime.now()).days / 365.0
            if T <= 0:
                return self._to_json(None, "Expiration date must be in the future.")

            chain = ticker.option_chain(expiration_date)
            options_df = chain.calls if option_type.lower() == "call" else chain.puts
            if options_df.empty:
                return self._to_json(None, f"No {option_type} chain for {expiration_date}.")

            hv = self._get_annualized_volatility(symbol, period=period)
            hv_pct = round(hv * 100, 2) if hv is not None and hv > 0 else None

            valid = options_df[
                options_df["impliedVolatility"].notna()
                & (options_df["impliedVolatility"] > 0)
                & options_df["lastPrice"].notna()
            ].copy()
            if valid.empty:
                return self._to_json(None, f"No valid IV/lastPrice for {expiration_date}.")

            valid["dist"] = (valid["strike"] - S).abs()
            valid = valid.sort_values("dist").head(max_strikes)

            rows = []
            for _, row in valid.iterrows():
                strike = float(row["strike"])
                sigma = float(row["impliedVolatility"])
                last_price = float(row["lastPrice"])
                d1 = (np.log(S / strike) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                d2 = d1 - sigma * np.sqrt(T)
                if option_type.lower() == "call":
                    bs_price = (S * norm.cdf(d1)) - (strike * np.exp(-r * T) * norm.cdf(d2))
                    delta = norm.cdf(d1)
                else:
                    bs_price = (strike * np.exp(-r * T) * norm.cdf(-d2)) - (S * norm.cdf(-d1))
                    delta = norm.cdf(d1) - 1

                iv_pct = round(sigma * 100, 2)
                opt = {
                    "strike": strike,
                    "last_price": round(last_price, 4),
                    "bs_price": round(bs_price, 4),
                    "iv_pct": iv_pct,
                    "hv_pct": hv_pct,
                    "delta": round(delta, 4),
                }
                if hv is not None and hv > 0:
                    iv_hv_ratio = round(sigma / hv, 2)
                    opt["iv_hv_ratio"] = iv_hv_ratio
                    opt["rich_cheap"] = "Rich" if iv_hv_ratio > 1.1 else ("Cheap" if iv_hv_ratio < 0.9 else "Fair")
                rows.append(opt)

            result = {
                "symbol": symbol,
                "expiration_date": expiration_date,
                "option_type": option_type,
                "spot_price": round(S, 2),
                "hv_pct": hv_pct,
                "options": rows,
            }
            return self._to_json(result)
        except Exception as e:
            return self._to_json(None, f"Error fetching options chain: {str(e)}")