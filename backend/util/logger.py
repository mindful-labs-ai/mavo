import logging
import os
from datetime import datetime
from termcolor import colored

class CustomLogger(logging.Logger):
    def _log_with_args(self, level, *args, **kwargs):
        message = ' '.join(map(str, args))
        # Apply color to messages for console output only, done in console formatter
        self.log(level, message, **kwargs)
        # if args:
        #     # message = args[0]  # The format string
        #     # args = args[1:]    # The arguments for the format string

        #     message = args[0] if isinstance(args[0], str) else " ".join(map(str, args))
        #     self.log(level, message, **kwargs)
        # else:
        #     message = ""
        # self.log(level, message, *args, **kwargs)

    def debug(self, *args, **kwargs):
        self._log_with_args(logging.DEBUG, *args, **kwargs)

    def info(self, *args, **kwargs):
        self._log_with_args(logging.INFO, *args, **kwargs)

    def warning(self, *args, **kwargs):
        self._log_with_args(logging.WARNING, *args, **kwargs)

    def error(self, *args, **kwargs):
        self._log_with_args(logging.ERROR, *args, **kwargs)

    def critical(self, *args, **kwargs):
        self._log_with_args(logging.CRITICAL, *args, **kwargs)

# Patch the logging module to use CustomLogger
logging.setLoggerClass(CustomLogger)

class ColoredFormatter(logging.Formatter):
    """Formatter that applies color codes to the output for console use only."""
    COLORS = {
        logging.DEBUG: 'blue',
        logging.INFO: 'green',
        logging.WARNING: 'yellow',
        logging.ERROR: 'red',
        logging.CRITICAL: 'magenta',
    }

    def format(self, record):
        if hasattr(record, 'levelname'):
            color = self.COLORS.get(record.levelno, 'white')
            record.msg = colored(record.msg, color)
        return super().format(record)

def setup_logging(console_verbosity=logging.INFO, file_verbosity=logging.DEBUG):
    log_dir = 'log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(log_dir, f'{timestamp}.log')
    
    print("Logging started to path", log_filename)

    # Create handlers for both file and console
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(file_verbosity)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_verbosity)
    console_formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(console_formatter)

    # Configure the logging
    logging.basicConfig(
        level=min(console_verbosity, file_verbosity),
        handlers=[file_handler, console_handler]
    )
    
    return log_filename

logger_instance = None
log_filename = None

def get_logger(name, console_verbosity=logging.INFO, file_verbosity=logging.DEBUG):
    global logger_instance, log_filename
    if not logger_instance:
        log_filename = setup_logging(console_verbosity, file_verbosity)
        logger_instance = logging.getLogger(name)
    return logger_instance

def get_log_filename():
    return log_filename

# Example usage
if __name__ == '__main__':
    logger = get_logger("exampleLogger")
    logger.info("This is an info message.")
    try:
        1 / 0  # Intentional error
    except ZeroDivisionError:
        logger.error("An error occurred", exc_info=True)  # Correctly logs the exception
    print("Log file is at:", get_log_filename())