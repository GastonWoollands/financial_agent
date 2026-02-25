import os
import json
import logging
import asyncio
import time
from telegram.helpers import escape_markdown
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler
# from textwrap import dedent
from agno.agent import Agent
from agents_utils import DEFAULT_RESPONSE, WELCOME_MESSAGE, PARSE_MODE
from commands import CommandConfig, COMMANDS, master_agent
from progress_indicator import ProgressIndicator
from logging.handlers import RotatingFileHandler

#----------------------------------------------------------------------------

# Configure logging
def setup_logging():
    """Configure logging with rotation and formatting."""
    os.makedirs('logs', exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Filter out httpx logs
    logging.getLogger('httpx').setLevel(logging.WARNING)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # File handler with rotation (10MB per file, keep 5 backup files)
    file_handler = RotatingFileHandler(
        'logs/bot.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to root logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logging
logger = setup_logging()

#----------------------------------------------------------------------------

def extract_symbol(text: str) -> str:
    """
    Extracts the stock symbol from a user's message.
    - If '$' is present, extracts the ticker after '$'.
    - If '$' is not present, uses the raw text as the ticker.
    """
    try:
        dollar_index = text.find("$")
        if dollar_index != -1 and dollar_index + 1 < len(text):
            ticker = text[dollar_index + 1:].split()[0].upper()
            logger.debug(f"Extracted symbol with $: {ticker}")
            return ticker
        else:
            ticker = text.strip().upper()
            logger.debug(f"Extracted symbol without $: {ticker}")
            return ticker
    except Exception as e:
        logger.error(f"Error extracting symbol from text '{text}': {str(e)}")
        return None


def parse_bs_args(args: list) -> tuple:
    """
    Parse /bs command arguments: symbol, strike, expiration_date, option_type.
    Returns (symbol, strike, expiration_date, option_type) on success,
    or (None, error_message) on failure.
    """
    if not args or len(args) != 4:
        return (
            None,
            "Uso: /bs $TICKER STRIKE YYYY-MM-DD call|put. Ejemplo: /bs $AAPL 200 2025-06-20 call",
        )
    symbol_raw, strike_str, expiration_date, option_type_raw = args
    symbol = extract_symbol(symbol_raw)
    if not symbol:
        return (None, "Uso: /bs $TICKER STRIKE YYYY-MM-DD call|put. Ejemplo: /bs $AAPL 200 2025-06-20 call")
    try:
        strike = float(strike_str)
        if strike <= 0:
            return (None, "El strike tiene que ser un número positivo.")
    except (ValueError, TypeError):
        return (None, "El strike tiene que ser un número positivo.")
    try:
        exp_date = datetime.strptime(expiration_date, "%Y-%m-%d")
        if exp_date.date() <= datetime.now().date():
            return (None, "La fecha de vencimiento tiene que ser YYYY-MM-DD y en el futuro.")
    except ValueError:
        return (None, "La fecha de vencimiento tiene que ser YYYY-MM-DD y en el futuro.")
    option_type = option_type_raw.strip().lower()
    if option_type not in ("call", "put"):
        return (None, "El tipo de opción tiene que ser 'call' o 'put'.")
    return (symbol, strike, expiration_date, option_type)


def parse_opciones_lista_args(args: list) -> tuple:
    """
    Parse /opciones_lista arguments: ticker, YYYY-MM-DD, call|put, [N].
    Returns (symbol, expiration_date, option_type, max_strikes) on success,
    or (None, error_message) on failure.
    """
    if not args or len(args) < 3:
        return (
            None,
            "Uso: /opciones_lista $TICKER YYYY-MM-DD call|put [N]. Ejemplo: /opciones_lista $AAPL 2025-06-20 call 10",
        )
    symbol_raw, expiration_date, option_type_raw = args[0], args[1], args[2]
    symbol = extract_symbol(symbol_raw)
    if not symbol:
        return (None, "Uso: /opciones_lista $TICKER YYYY-MM-DD call|put [N]. Ejemplo: /opciones_lista $AAPL 2025-06-20 call 10")
    try:
        exp_date = datetime.strptime(expiration_date, "%Y-%m-%d")
        if exp_date.date() <= datetime.now().date():
            return (None, "La fecha de vencimiento tiene que ser YYYY-MM-DD y en el futuro.")
    except ValueError:
        return (None, "La fecha de vencimiento tiene que ser YYYY-MM-DD.")
    option_type = option_type_raw.strip().lower()
    if option_type not in ("call", "put"):
        return (None, "El tipo de opción tiene que ser 'call' o 'put'.")
    max_strikes = 10
    if len(args) >= 4:
        try:
            max_strikes = int(args[3])
            if max_strikes < 1 or max_strikes > 20:
                return (None, "N tiene que ser un número entre 1 y 20.")
        except (ValueError, TypeError):
            return (None, "N tiene que ser un número entre 1 y 20.")
    return (symbol, expiration_date, option_type, max_strikes)


#----------------------------------------------------------------------------

async def get_agent_response(
    agent: Agent,
    query: str,
    progress: ProgressIndicator = None,
    user_id: int | None = None,
) -> str:
    """Runs a query through the financial agent and returns the response."""
    try:
        logger.info(f"Running agent query: {query}")
        if progress:
            await progress.update_text("Consultando datos financieros")

        session_id = f"telegram-{user_id}" if user_id is not None else None

        response = agent.run(
            input=query,
            user_id=str(user_id) if user_id is not None else None,
            session_id=session_id,
            add_history_to_context=True,
        )
        response_content = response.content if hasattr(response, "content") else str(response)
        logger.debug(f"Agent response received: {response_content[:100]}...")
        return response_content
    except Exception as e:
        logger.error(f"Error running agent query '{query}': {str(e)}")
        return DEFAULT_RESPONSE

#----------------------------------------------------------------------------

# Add conversation states
WAITING_FOR_SYMBOL = 1

# Add rate limiting
from datetime import datetime, timedelta
from collections import defaultdict

class RateLimiter:
    def __init__(self, max_requests: int, time_window: int):
        self.max_requests = max_requests
        self.time_window = time_window
        self.user_requests = defaultdict(list)
        logger.info(f"Rate limiter initialized: {max_requests} requests per {time_window} seconds")
    
    def is_allowed(self, user_id: int) -> bool:
        now = datetime.now()
        user_timestamps = self.user_requests[user_id]
        user_timestamps = [ts for ts in user_timestamps if now - ts < timedelta(seconds=self.time_window)]
        self.user_requests[user_id] = user_timestamps
        
        if len(user_timestamps) >= self.max_requests:
            logger.warning(f"Rate limit exceeded for user {user_id}")
            return False
        
        user_timestamps.append(now)
        logger.debug(f"Request allowed for user {user_id}, current count: {len(user_timestamps)}")
        return True

rate_limiter = RateLimiter(max_requests=10, time_window=60)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a help message with all available commands."""
    user_id = update.effective_user.id
    logger.info(f"Help command requested by user {user_id}")
    help_text = """📊 Cómo hablar conmigo:

Podés escribirme en texto libre (sin comandos) para charlar sobre mercados, acciones, opciones y riesgo. Yo uso datos en tiempo real y mis herramientas financieras para responder.

Comandos disponibles (atajos):

/precio - Te tiro el precio de una acción. Ejemplo: /precio $AAPL
/noticias - Las últimas novedades de una empresa. Ejemplo: /noticias $TSLA
/noticias_general - Un repasito rápido de cómo viene la mano en el mercado.
/tecnicos - Análisis técnico del activo que gustes. Ejemplo: /tecnicos $GOOGL
/fundamentales - Los números pesados de una empresa. Ejemplo: /fundamentales $AAPL
/correlacion - Te cuento cómo se llevan una lista de acciones. Ejemplo: /correlacion $AAPL $MELI
/volatilidad - Te analizo la volatilidad de una acción. Ejemplo: /volatilidad $MELI
/opciones - Te analizo opciones financieras de una acción. Ejemplo: /opciones $MELI
/opciones_lista - Lista de opciones con BS, IV, HV y rich/cheap. Ejemplo: /opciones_lista $AAPL 2025-06-20 call
/bs - Precio teórico Black-Scholes de una opción. Ejemplo: /bs $AAPL 200 2025-06-20 call

Opciones: IV vs HV: >1.1 rich, <0.9 cheap.

💡 Tips:
Usá MAYÚSCULAS para los tickers
Podés usar el símbolo $ o no
Para más info, usá /ayuda"""

    await update.message.reply_text(help_text)
    logger.debug(f"Help message sent to user {user_id}")


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors in the bot."""
    user_id = update.effective_user.id if update and update.effective_user else "unknown"
    error = context.error
    
    logger.error(f"Error for user {user_id}: {str(error)}")
    
    if isinstance(error, ValueError):
        error_msg = "Che, algo salió mal con los datos. Asegurate de usar un ticker válido."
    elif isinstance(error, TimeoutError):
        error_msg = "Se me colgó la conexión. Intentá de nuevo en un ratito."
    else:
        error_msg = "Ups, algo salió mal. Intentá de nuevo más tarde."
    
    if update and update.message:
        await update.message.reply_text(error_msg)
        logger.debug(f"Error message sent to user {user_id}")

def log_bot_response(user_id: int, command: str, response: str, execution_time: float):
    """Log the bot's response to a user query."""
    logger.info(f"Bot response for user {user_id} - Command: {command}")
    logger.info(f"Response: {response[:200]}..." if len(response) > 200 else f"Response: {response}")
    logger.info(f"Execution time: {execution_time:.2f} seconds")

async def handle_command(update: Update, context: ContextTypes.DEFAULT_TYPE, config: CommandConfig) -> None:
    """Generic handler for bot commands supporting variable symbol counts."""
    user_id = update.effective_user.id
    command = update.message.text.split()[0]
    start_time = time.time()
    
    logger.info(f"Command '{command}' received from user {user_id}")
    
    # Check rate limit
    if not rate_limiter.is_allowed(user_id):
        logger.warning(f"Rate limit exceeded for user {user_id}")
        await update.message.reply_text("Che, estás haciendo muchas consultas. Esperá un minuto y volvé a intentar.")
        return
    
    # Initialize progress indicator
    progress = ProgressIndicator(update, context)
    await progress.start("Iniciando análisis")
    
    try:
        if config.agent is None:  # Special case for /start and /help
            logger.debug(f"Handling special command: {command} for user {user_id}")
            if update.message.text.startswith("/help"):
                await progress.stop()
                await help_command(update, context)
            else:
                await progress.stop()
                await update.message.reply_text(WELCOME_MESSAGE)
            return

        query = None
        args = " ".join(context.args) if context.args else ""
        symbols = [extract_symbol(arg) for arg in context.args if extract_symbol(arg)] if args else []

        logger.info(f"Command: {config.description}, User: {user_id}, Symbols: {symbols}")

        # Validate symbol count
        if config.requires_symbol:
            if len(symbols) < config.required_symbols_min:
                message = f"Che, mandaste pocos tickers. Necesito al menos {config.required_symbols_min}. Ejemplo: /{update.message.text.split()[0][1:]} {' '.join(['$SYM'] * config.required_symbols_min)}"
                logger.warning(f"Invalid symbol count for user {user_id}: {len(symbols)} < {config.required_symbols_min}")
                await progress.stop()
                await update.message.reply_text(message)
                return
            if config.required_symbols_max and len(symbols) > config.required_symbols_max:
                message = f"Che, mandaste demasiados tickers. Máximo {config.required_symbols_max}."
                logger.warning(f"Too many symbols for user {user_id}: {len(symbols)} > {config.required_symbols_max}")
                await progress.stop()
                await update.message.reply_text(message)
                return
        
        if config.query_template:
            try:
                if config.required_symbols_min == 0 and not symbols:  # Handle /noticias without ticker
                    query = config.query_template
                    logger.debug(f"Using template without symbols for user {user_id}")

                elif config.required_symbols_min == 1 and len(symbols) == 1:
                    query = config.query_template.format(symbol=symbols[0])
                    logger.debug(f"Formatted query with single symbol: {symbols[0]} for user {user_id}")

                elif config.required_symbols_min >= 2 and len(symbols) >= 2:
                    query = config.query_template.format(symbols=" ".join(symbols))
                    logger.debug(f"Formatted query with multiple symbols: {symbols} for user {user_id}")

                else:
                    query = config.query_template  # Fallback
                    logger.debug(f"Using template as fallback for user {user_id}")

                logger.info(f"Executing query for user {user_id}: {query}")
                await progress.update_text("Procesando datos")
                response_text = await get_agent_response(
                    config.agent,
                    query,
                    progress,
                    user_id=user_id,
                )
                logger.debug(f"Response length for user {user_id}: {len(response_text)} characters")

            except IndexError:
                logger.error(f"Index error processing symbols for user {user_id}: {symbols}")
                response_text = "Che, algo salió mal con los tickers. Asegurate de mandarlos bien."
            except KeyError as e:
                logger.error(f"Key error in template formatting for user {user_id}: {str(e)}")
                response_text = f"Error en el formato: {str(e)}. Usá el ejemplo del comando."
        else:
            logger.warning(f"No query template defined for command: {command} from user {user_id}")
            response_text = "Comando no implementado correctamente, che."

        # Stop progress indicator and send response
        await progress.stop()
        await update.message.reply_text(response_text)
        
        # Log the bot's response
        execution_time = time.time() - start_time
        log_bot_response(user_id, command, response_text, execution_time)

    except Exception as e:
        logger.error(f"Unexpected error in handle_command for user {user_id}: {str(e)}")
        await progress.stop()
        await update.message.reply_text("Ups, algo salió mal. Intentá de nuevo más tarde.")

#----------------------------------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await handle_command(update, context, COMMANDS["start"])

#----------------------------------------------------------------------------

async def price(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await handle_command(update, context, COMMANDS["precio"])

#----------------------------------------------------------------------------

async def news(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await handle_command(update, context, COMMANDS["noticias"])

#----------------------------------------------------------------------------

async def news_general(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await handle_command(update, context, COMMANDS["noticias_general"])

#----------------------------------------------------------------------------

async def technical_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await handle_command(update, context, COMMANDS["tecnicos"])

#----------------------------------------------------------------------------

async def fundamental_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await handle_command(update, context, COMMANDS["fundamentales"])

#----------------------------------------------------------------------------

async def correlation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await handle_command(update, context, COMMANDS["correlacion"])

#----------------------------------------------------------------------------

async def volatility(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await handle_command(update, context, COMMANDS["volatilidad"])

#----------------------------------------------------------------------------

async def options(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await handle_command(update, context, COMMANDS["opciones"])


#----------------------------------------------------------------------------

async def bs_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /bs $TICKER STRIKE YYYY-MM-DD call|put for Black-Scholes pricing."""
    user_id = update.effective_user.id
    command = "/bs"
    start_time = time.time()

    logger.info(f"Command '{command}' received from user {user_id}")

    if not rate_limiter.is_allowed(user_id):
        logger.warning(f"Rate limit exceeded for user {user_id}")
        await update.message.reply_text("Che, estás haciendo muchas consultas. Esperá un minuto y volvé a intentar.")
        return

    progress = ProgressIndicator(update, context)
    await progress.start("Calculando Black-Scholes")

    try:
        parsed = parse_bs_args(context.args or [])
        if parsed[0] is None:
            await progress.stop()
            await update.message.reply_text(parsed[1])
            return

        symbol, strike, expiration_date, option_type = parsed
        query = (
            f"Calculate the Black-Scholes theoretical price for {symbol}, strike {strike}, "
            f"expiration date {expiration_date}, {option_type} option. "
            f"Use get_black_scholes_pricing with symbol={symbol}, strike={strike}, "
            f"expiration_date={expiration_date}, option_type={option_type}. "
            "Respond with a short bullet-point list only. Include: current underlying price, "
            "last option price, Black-Scholes theoretical price, delta, market implied volatility (IV). "
            "When the tool returns them, also include HV (1y) and IV vs HV (or IV/HV ratio and Rich/Fair/Cheap) so the user can see if the option is overpriced. "
            "Use bullet points (e.g. - or •), no long paragraphs."
        )
        logger.info(f"BS query for user {user_id}: {query}")
        await progress.update_text("Procesando datos")
        config = COMMANDS["bs"]
        response_text = await get_agent_response(
            config.agent,
            query,
            progress,
            user_id=user_id,
        )
        await progress.stop()
        await update.message.reply_text(response_text)
        execution_time = time.time() - start_time
        log_bot_response(user_id, command, response_text, execution_time)
    except Exception as e:
        logger.error(f"Unexpected error in bs_command for user {user_id}: {str(e)}")
        await progress.stop()
        await update.message.reply_text("Ups, algo salió mal. Intentá de nuevo más tarde.")


async def opciones_lista_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /opciones_lista $TICKER YYYY-MM-DD call|put [N] for options chain with metrics."""
    user_id = update.effective_user.id
    command = "/opciones_lista"
    start_time = time.time()

    logger.info(f"Command '{command}' received from user {user_id}")

    if not rate_limiter.is_allowed(user_id):
        logger.warning(f"Rate limit exceeded for user {user_id}")
        await update.message.reply_text("Che, estás haciendo muchas consultas. Esperá un minuto y volvé a intentar.")
        return

    progress = ProgressIndicator(update, context)
    await progress.start("Listando opciones")

    try:
        parsed = parse_opciones_lista_args(context.args or [])
        if parsed[0] is None:
            await progress.stop()
            await update.message.reply_text(parsed[1])
            return

        symbol, expiration_date, option_type, max_strikes = parsed
        query = (
            f"Get a list of options for {symbol}, expiration {expiration_date}, {option_type}. "
            f"Use get_options_chain_with_metrics with symbol={symbol}, expiration_date={expiration_date}, "
            f"option_type={option_type}, max_strikes={max_strikes}. "
            "Respond with a short, readable table: Strike, Last, BS, IV%, HV%, IV/HV or Rich/Fair/Cheap, Delta. "
            "Use a compact format that fits Telegram. Add one line: IV > HV means rich, IV < HV means cheap."
        )
        logger.info(f"opciones_lista query for user {user_id}: {query}")
        await progress.update_text("Procesando datos")
        config = COMMANDS["opciones_lista"]
        response_text = await get_agent_response(
            config.agent,
            query,
            progress,
            user_id=user_id,
        )
        await progress.stop()
        await update.message.reply_text(response_text)
        execution_time = time.time() - start_time
        log_bot_response(user_id, command, response_text, execution_time)
    except Exception as e:
        logger.error(f"Unexpected error in opciones_lista_command for user {user_id}: {str(e)}")
        await progress.stop()
        await update.message.reply_text("Ups, algo salió mal. Intentá de nuevo más tarde.")


async def conversation_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle open-text conversational messages (no leading command)."""
    user_id = update.effective_user.id
    command = "conversation_message"
    start_time = time.time()

    logger.info(f"Free-text message from user {user_id}")

    if not rate_limiter.is_allowed(user_id):
        logger.warning(f"Rate limit exceeded for user {user_id}")
        await update.message.reply_text(
            "Estás haciendo muchas consultas. Esperá un minuto y volvé a intentar."
        )
        return

    progress = ProgressIndicator(update, context)
    await progress.start("Pensando tu consulta financiera")

    try:
        user_query = (update.message.text or "").strip()
        if not user_query:
            await progress.stop()
            await update.message.reply_text(
                "Contame qué querés saber del mercado, de un ticker o de tu estrategia."
            )
            return

        prompt = (
            "Modo conversación abierta sobre mercados y finanzas. "
            "Usá las herramientas financieras para responder con datos actuales.\n\n"
            f"Usuario: {user_query}"
        )

        logger.info(f"Conversation prompt for user {user_id}: {prompt}")
        await progress.update_text("Consultando datos y armando respuesta")

        response_text = await get_agent_response(
            master_agent,
            prompt,
            progress,
            user_id=user_id,
        )

        await progress.stop()
        await update.message.reply_text(response_text)

        execution_time = time.time() - start_time
        log_bot_response(user_id, command, response_text, execution_time)

    except Exception as e:
        logger.error(f"Unexpected error in conversation_message for user {user_id}: {str(e)}")
        await progress.stop()
        await update.message.reply_text(
            "Ups, algo salió mal. Intentá de nuevo más tarde."
        )

#----------------------------------------------------------------------------

def setup_application() -> ApplicationBuilder:
    """Initialize and configure the Telegram application."""
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        logger.error("TELEGRAM_TOKEN not found in environment variables")
        raise ValueError("Telegram token not found.")
    logger.info("Telegram application setup completed")
    return ApplicationBuilder().token(token)

#----------------------------------------------------------------------------

def register_handlers(app):
    """Register command handlers with the application."""
    logger.info("Registering command handlers")
    
    # Add error handler
    app.add_error_handler(error_handler)
    logger.debug("Error handler registered")
    
    # Add help command handler
    app.add_handler(CommandHandler("help", help_command))
    logger.debug("Help command handler registered")
    
    # Register other command handlers
    handlers = {
        "start": start,
        "precio": price,
        "noticias": news,
        "noticias_general": news_general,
        "tecnicos": technical_analysis,
        "fundamentales": fundamental_analysis,
        "correlacion": correlation,
        "volatilidad": volatility,
        "opciones": options,
        "opciones_lista": opciones_lista_command,
        "bs": bs_command,
    }
    for command, handler in handlers.items():
        app.add_handler(CommandHandler(command, handler))
        logger.debug(f"Handler registered for command: {command}")

    # Free-text conversational handler (no leading slash)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, conversation_message))
    logger.debug("Free-text conversation handler registered")

    logger.info("All handlers registered successfully")

#----------------------------------------------------------------------------

def main():
    """Initialize and run the bot."""
    logger.info("Starting bot initialization")
    app = setup_application().build()
    register_handlers(app)
    logger.info("Bot started and running...")
    app.run_polling()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()