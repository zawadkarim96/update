import pytest
from strategy_modifier import add_new_strategy, remove_strategy
import os

def test_add_remove(tmp_path):
    file = tmp_path / "test_strategies.py"
    file.write_text("strategies = {}\ndef existing(df): pass")
    
    add_new_strategy(str(file), "test_strategy", "def test_strategy(df): return 0")
    assert "def test_strategy" in file.read_text()
    
    remove_strategy(str(file), "test_strategy")
    assert "def test_strategy" not in file.read_text()