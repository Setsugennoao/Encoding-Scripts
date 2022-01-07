import math
from os import path

COMMON_MASKS_PATH = path.join(path.dirname(__file__), r".\masks")

grain_amount = math.pi / 20 + 69 / 420 / 21 * 9

degrain_args = [(500, 750), (1550, 750), (1300, 750)]

light_knl_args = dict(trange=[2, 1], search_radius=[4, 2], similarity_radius=8, sigma=[2.2, 1.5])

heavy_knl_args = dict(trange=[2, 1], search_radius=[4, 2], similarity_radius=[8, 4], sigma=[6, 2.75, 2.3])
