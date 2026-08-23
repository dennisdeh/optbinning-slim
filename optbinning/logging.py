"""
Logging class.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import logging
import sys


class Logger:
    def __init__(self, logger_name=None, filename=None):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s : %(message)s')

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        if filename is not None:
            fhandler = logging.FileHandler(filename)
            fhandler.setFormatter(formatter)
            self.logger.addHandler(fhandler)

    def close(self):
        # Iterate a copy: whether removeHandler rebinds logger.handlers or
        # mutates it in place varies by CPython micro version (gh-79366), and
        # iterating the live list skips every second handler where it mutates.
        # See reports/DECISIONS.md.
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)
