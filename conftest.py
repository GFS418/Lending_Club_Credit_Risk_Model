"""Make src/ importable in tests (data_prep, lc_data_dictionary)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
