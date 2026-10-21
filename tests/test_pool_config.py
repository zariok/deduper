"""Tests for the env-driven worker pool sizes in deduper.config.

Config reads the environment once, at import, so these exercise the parsing
helper directly rather than the class attributes.
"""

import logging

import pytest

from deduper import config as config_module
from deduper.config import default_hash_workers, env_positive_int

VAR = "DEDUPER_TEST_POOL_SIZE"


class TestEnvPositiveInt:
    def test_unset_uses_default(self, monkeypatch):
        monkeypatch.delenv(VAR, raising=False)
        assert env_positive_int(VAR, 7) == 7

    def test_blank_uses_default(self, monkeypatch):
        monkeypatch.setenv(VAR, "  ")
        assert env_positive_int(VAR, 7) == 7

    def test_set_value_is_used(self, monkeypatch):
        monkeypatch.setenv(VAR, "32")
        assert env_positive_int(VAR, 7) == 32

    def test_surrounding_whitespace_is_tolerated(self, monkeypatch):
        monkeypatch.setenv(VAR, " 12\n")
        assert env_positive_int(VAR, 7) == 12

    @pytest.mark.parametrize("raw", ["abc", "4.5", "0", "-3", "8 workers"])
    def test_invalid_falls_back_with_warning(self, monkeypatch, caplog, raw):
        monkeypatch.setenv(VAR, raw)
        with caplog.at_level(logging.WARNING, logger=config_module.__name__):
            assert env_positive_int(VAR, 7) == 7
        assert any(VAR in record.getMessage() for record in caplog.records)


class TestDefaultHashWorkers:
    def test_half_the_cores(self, monkeypatch):
        monkeypatch.setattr(config_module.os, "cpu_count", lambda: 80)
        assert default_hash_workers() == 40

    def test_never_below_four(self, monkeypatch):
        monkeypatch.setattr(config_module.os, "cpu_count", lambda: 2)
        assert default_hash_workers() == 4

    def test_unknown_core_count(self, monkeypatch):
        monkeypatch.setattr(config_module.os, "cpu_count", lambda: None)
        assert default_hash_workers() == 4


def test_config_exposes_pool_sizes():
    from deduper.config import Config

    for name in ("HASH_WORKERS", "FFMPEG_WORKERS", "SCAN_WORKERS"):
        value = getattr(Config, name)
        assert isinstance(value, int) and value >= 1
