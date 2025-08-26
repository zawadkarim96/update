"""Demo script used in the original project.

It does not form part of the automated test suite, so we mark the module
to be skipped when ``pytest`` collects tests.
"""

import pytest

pytest.skip("Demo script – skipped during automated tests", allow_module_level=True)