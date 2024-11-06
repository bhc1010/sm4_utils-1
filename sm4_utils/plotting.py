import os
from . import formats

def plot_waterfall(src: str):
    name, ext = os.path.splitext(src)
    if not ext:
        raise FileNotFoundError("File source must be a valid format.")

    match ext:
        case '.sm4':
            sm4 = formats.SM4(src)
            sm4.plot_waterfall()
        case _:
            pass