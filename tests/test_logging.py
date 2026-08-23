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

    # both handlers, not just the first. See reports/DECISIONS.md.
    assert not len(logger.logger.handlers)


def test_close_when_removehandler_mutates_in_place(tmp_path, monkeypatch):
    # Whether logging.Logger.removeHandler rebinds handlers or mutates it in
    # place varies by CPython micro version (gh-79366): 3.13.15 and 3.14.7
    # rebind, 3.14.6 — the macOS CI runner on 2026-08-23 — mutates. close()
    # must remove every handler either way, so pin the mutating semantics
    # explicitly rather than only whichever the local interpreter happens to
    # implement. See reports/DECISIONS.md.
    def removeHandler(self, hdlr):
        if hdlr in self.handlers:
            self.handlers.remove(hdlr)

    monkeypatch.setattr(logging.Logger, "removeHandler", removeHandler)

    path = tmp_path / "optbinning_mutating.log"
    logger = Logger("test_close_mutating", filename=str(path))

    assert len(logger.logger.handlers) == 2

    logger.close()

    assert not len(logger.logger.handlers)
