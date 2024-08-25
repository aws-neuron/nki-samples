import os
import sys

# This file is basically a hack around the fact that pytest has a bug where it does not discover conftest.py correctly if you launch the test using --pyargs.
# https://github.com/pytest-dev/pytest/issues/1596


# Todo: Using __file__ isn't the most robust. Figure out how to do this using importlib or similar.
test_root = os.path.dirname(__file__)

if __name__ == "__main__":
  import pytest
  errcode = pytest.main([test_root] + sys.argv[1:])
  sys.exit(errcode)