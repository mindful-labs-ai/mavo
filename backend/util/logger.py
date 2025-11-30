import logging


def setup_logging(console_verbosity=logging.INFO, file_verbosity=logging.DEBUG):
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_verbosity)

    # Configure the logging
    logging.basicConfig(
        level=min(console_verbosity, file_verbosity), handlers=[console_handler]
    )


logger_instance = None


def get_logger(name, console_verbosity=logging.INFO, file_verbosity=logging.DEBUG):
    global logger_instance
    if not logger_instance:
        setup_logging(console_verbosity, file_verbosity)
        logger_instance = logging.getLogger(name)
    return logger_instance


if __name__ == "__main__":
    logger = get_logger("exampleLogger")
    logger.info("This is an info message.")
    try:
        1 / 0  # Intentional error
    except ZeroDivisionError:
        logger.error("An error occurred", exc_info=True)  # Correctly logs the exception
