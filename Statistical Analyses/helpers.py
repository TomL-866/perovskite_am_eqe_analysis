import os


def get_base_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
