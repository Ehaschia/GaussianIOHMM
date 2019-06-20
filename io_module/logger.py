_author__ = 'Ehaschia'

import logging
import sys


def get_logger(name, log_dir, level=logging.INFO, handler=sys.stdout,
               formatter='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(formatter)

    fh = logging.FileHandler(log_dir + 'info.log')
    fh.setLevel(level)
    fh.setFormatter(formatter)

    stream_handler = logging.StreamHandler(handler)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(fh)
    return logger
