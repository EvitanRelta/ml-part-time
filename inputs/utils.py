import os
from typing import Callable


def set_abs_path_to(current_dir: str) -> Callable[[str], str]:
    return lambda path: os.path.join(current_dir, path)
