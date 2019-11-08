__author__ = 'Ehaschia'

import logging
import sys
import os


def get_logger(name, level=logging.INFO, handler=sys.stdout,
               formatter='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(formatter)

    stream_handler = logging.StreamHandler(handler)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    return logger


def close_dir_logger(logger):
    if len(logger.handlers) < 2:
        return
    fh = logger.handlers[1]
    fh.close()
    logger.removeHandler(fh)


def change_handler(logger, log_dir, level=logging.INFO,
                   formatter='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    close_dir_logger(logger)
    formatter = logging.Formatter(formatter)
    fh = logging.FileHandler(log_dir + 'info.log')
    fh.setLevel(level)
    fh.setFormatter(formatter)

    logger.addHandler(fh)
