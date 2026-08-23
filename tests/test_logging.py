"""
Logging class testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import logging

from optbinning.logging import Logger


def test_stream_handler():
    logger = Logger("test_stream_handler")

    assert len(logger.logger.handlers) == 1
    assert isinstance(logger.logger.handlers[0], logging.StreamHandler)


def test_file_handler(tmp_path):
    path = tmp_path / "optbinning.log"
    logger = Logger("test_file_handler", filename=str(path))

    assert len(logger.logger.handlers) == 2

    logger.logger.info("a message")

    assert "a message" in path.read_text()


def test_close(tmp_path):
    path = tmp_path / "optbinning_close.log"
    logger = Logger("test_close", filename=str(path))

    logger.close()

    # both handlers, not just the first: removeHandler replaces the handler
    # list rather than mutating it, so the loop in close() is not iterating
    # the list it shrinks. See reports/DECISIONS.md.
    assert not len(logger.logger.handlers)
