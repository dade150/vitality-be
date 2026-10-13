import sys

from loguru import logger


def configure_logging(is_prod: bool) -> None:
    """Loguru sicuro. `diagnose=False` e `backtrace=False`: altrimenti le traceback stampano
    i VALORI delle variabili (chat, token). Si applica anche in dev, per abitudine sicura."""
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO" if is_prod else "DEBUG",
        diagnose=False,
        backtrace=False,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | rid={extra[request_id]} | {message}",
        filter=lambda r: r["extra"].setdefault("request_id", "-") or True,
    )
