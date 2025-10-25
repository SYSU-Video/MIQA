import logging
from typing import Optional
from pathlib import Path


def build_logger(
        output_dir: str,
        log_file: str,
        rank: Optional[int] = 0,
        level: int = logging.INFO,
        console_level: int = logging.INFO,
        file_level: int = logging.INFO
) -> logging.Logger:
    """Setup logging configuration for distributed training.

    Args:
        output_dir: Directory to store log files
        log_file: Name of the log file
        rank: Process rank in distributed training (default: 0)
        level: Overall logging level (default: logging.INFO)
        console_level: Console logging level (default: logging.INFO)
        file_level: File logging level (default: logging.INFO)

    Returns:
        logging.Logger: Configured logger instance

    Raises:
        OSError: If log directory creation fails
    """
    # Create log directory
    log_dir = Path(output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    # Add rank to log filename for distributed training
    if rank is not None and rank >= 0:
        filename = log_dir / f"{Path(log_file).stem}_rank{rank}{Path(log_file).suffix}"
    else:
        filename = log_dir / log_file

    # Get logger instance
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Logging format
    log_format = '[%(asctime)s %(levelname)-8s] %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(log_format, datefmt=date_format)

    # File handler
    try:
        file_handler = logging.FileHandler(filename, mode='w', encoding='utf-8')
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.error(f"Failed to create file handler: {str(e)}")
        raise

    # Console handler (only for rank 0 in distributed training)
    if rank in [0, None]:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Log initial information
    logger.info(f"Logging setup completed. Log file: {filename}")
    logger.info(f"Logging level - Overall: {logging.getLevelName(level)}, "
                f"Console: {logging.getLevelName(console_level)}, "
                f"File: {logging.getLevelName(file_level)}")

    return logger

