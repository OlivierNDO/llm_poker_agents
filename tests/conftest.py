# tests/conftest.py
import sys
import os

# add the project/src directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
