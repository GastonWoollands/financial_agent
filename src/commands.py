from dataclasses import dataclass
from agents_utils import create_financial_agent

#----------------------------------------------------------------------------

agents = {
    "general"             : create_financial_agent("general"),
    "technical"           : create_financial_agent("technical"),
    "fundamental"         : create_financial_agent("fundamental"),
    "pairs_and_volatility": create_financial_agent("pairs_and_volatility"),
    "options"             : create_financial_agent("options")
}

#----------------------------------------------------------------------------

@dataclass
class CommandConfig:
    """Configuration for a bot command."""
    description: str
    agent: object
    query_template: str
    requires_symbol: bool = True
    required_symbols_count: int = 1
    required_symbols_min: int = 1
    required_symbols_max: int = None
    no_args_message: str = "Por favor, especifica un símbolo válido. Ejemplo: /{} $AAPL"

#----------------------------------------------------------------------------

# Command configurations
COMMANDS = {
    "start": CommandConfig(
        description="Welcome message",
        agent=None,
        query_template="",
        requires_symbol=False,
        required_symbols_min=0
    ),
    "help": CommandConfig(
        description="Show help message with all available commands",
        agent=None,
        query_template="",
        requires_symbol=False,
        required_symbols_min=0
    ),
    "precio": CommandConfig(
        description="Get current stock price",
        agent=agents["general"],
        query_template="Dame el precio actual del ticker (symbol): {symbol}",
        no_args_message="Por favor, especificá el símbolo de la acción. Ejemplo: /precio $AAPL"
    ),
    "noticias": CommandConfig(
        description="Get company or market news",
        agent=agents["general"],
        query_template="Dame un resumen de las últimas noticias del ticker (symbol): {symbol}",
        requires_symbol=True,
        required_symbols_min=1,
        no_args_message=None
    ),

    "noticias_general": CommandConfig(
        description="Get company or market news",
        agent=agents["general"],
        query_template="Dame un resumen de las noticias del mercado financiero del ticker: ^GSPC y el ticker ^SPX",
        requires_symbol=False,
        required_symbols_min=0,
        no_args_message=None
    ),
    "tecnicos": CommandConfig(
        description="Get technical analysis",
        agent=agents["technical"],
        query_template="""Dame el resumen de los análisis técnicos del ticker: {symbol} usando los siguientes parámetros:
        - symbol: {symbol}
        - period: 1y
        - interval: 1d""",
        requires_symbol=True,
        required_symbols_min=1
    ),
    "fundamentales": CommandConfig(
        description="Get fundamental analysis",
        agent=agents["fundamental"],
        query_template="""Dame el resumen detallado de análisis fundamental del ticker: {symbol} usando los siguientes parámetros:
        - symbol: {symbol}
        - period: 1y
        - interval: 1d""",
        requires_symbol=True,
        required_symbols_min=1
    ),
    "correlacion": CommandConfig(
        description="Get correlation between two assets",
        agent=agents["pairs_and_volatility"],
        query_template="""Dame la correlacion entre los siguientes tickers: {symbols} usando los siguientes parámetros:
        - period: 1y
        - interval: 1d
        Incluye métricas de volatilidad, ratio de Sharpe, drawdown máximo, beta y potencial alcista.""",
        requires_symbol=True,
        required_symbols_min=2,
        no_args_message="Mandáme dos tickers. Ejemplo: /correlacion $^SPY $AAPL"
    ),
    "volatilidad": CommandConfig(
        description="Get volatility for the asset using the ^SPX as benchmark and the default period of 1 year",
        agent=agents["pairs_and_volatility"],
        query_template="""Dame un análisis de volatilidad para el ticker {symbol} usando los siguientes parámetros:
        - benchmark: ^SPX
        - period: 1y
        - interval: 1d
        Incluye métricas de volatilidad, ratio de Sharpe, drawdown máximo, beta y potencial alcista.""",
        requires_symbol=True,
        required_symbols_min=1,
        no_args_message="Mandáme un ticker. Ejemplo: /volatilidad $AAPL"
    ),
    "opciones": CommandConfig(
        description="Get options analysis for the asset using the default period of 1 year",
        agent=agents["options"],
        query_template="Dame analisis de opciones del ticker: {symbol} con los parametros por defecto",
        requires_symbol=True,
        required_symbols_min=1,
        no_args_message="Mandáme un ticker. Ejemplo: /opciones $AAPL"
    ),
    "opciones_lista": CommandConfig(
        description="List of options with BS, IV, HV and rich/cheap for an expiration",
        agent=agents["options"],
        query_template="",
        requires_symbol=True,
        required_symbols_min=1,
        no_args_message="Uso: /opciones_lista $TICKER YYYY-MM-DD call|put [N]. Ejemplo: /opciones_lista $AAPL 2025-06-20 call 10"
    ),
    "bs": CommandConfig(
        description="Black-Scholes theoretical option price",
        agent=agents["options"],
        query_template="",
        requires_symbol=True,
        required_symbols_min=1,
        no_args_message="Uso: /bs $TICKER STRIKE YYYY-MM-DD call|put. Ejemplo: /bs $AAPL 200 2025-06-20 call"
    ),
}