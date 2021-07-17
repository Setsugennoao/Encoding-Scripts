import math
from os import path

COMMON_MASKS_PATH = path.join(path.dirname(__file__), r".\masks")

grain_amount = math.pi / 20 + 69 / 420 / 21 * 9

degrain_args = [(500, 750), (1550, 750), (1300, 750)]

light_knl_args = [
    dict(h=2.2, a=4, d=2, s=8),
    dict(h=1.5, a=2, d=2, s=8),
    dict(h=1.5, a=2, d=2, s=8),
]

heavy_knl_args = [
    dict(h=6, a=4, d=2, s=8),
    dict(h=2.75, a=2, d=1, s=4),
    dict(h=2.3, a=2, d=1, s=4),
]
