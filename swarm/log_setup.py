import logging
from concurrent_log_handler import ConcurrentRotatingFileHandler


def setup_logging(log_path, log_level):
    # Use a dictionary to map log level strings to log level constants
    log_levels = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }

    # Set up logger
    logger = logging.getLogger()
    logger.setLevel(log_levels.get(log_level, logging.INFO))

    # Set up file handler
    handler = ConcurrentRotatingFileHandler(
        log_path, "a", maxBytes=50 * 1024 * 1024, backupCount=7
    )
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
        )
    )

    logger.addHandler(handler)
