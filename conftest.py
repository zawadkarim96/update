import pytest
from unittest.mock import patch

class Mocker:
    def __init__(self):
        self._patches = []
    def patch(self, target, *args, **kwargs):
        p = patch(target, *args, **kwargs)
        mocked = p.start()
        self._patches.append(p)
        return mocked
    def stopall(self):
        for p in self._patches:
            p.stop()
        self._patches.clear()

@pytest.fixture
def mocker():
    m = Mocker()
    yield m
    m.stopall()
