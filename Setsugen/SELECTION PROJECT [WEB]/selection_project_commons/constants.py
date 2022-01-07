import random
from vardefunc.noise import AddGrain

seed = random.seed()

graigasm_args = dict(
    thrs=[x << 8 for x in (26, 75, 130, 180)],
    strengths=[(0.7, 0.2), (1, 0.12), (1.05, 0.05), (0.12, 0)],
    sizes=(1.1, 1.53, 1.86, 1.7),
    sharps=(80, 60, 40, 40),
    grainers=[
        AddGrain(seed=seed, constant=False),
        AddGrain(seed=seed, constant=False),
        AddGrain(seed=seed, constant=False),
        AddGrain(seed=seed, constant=True)
    ]
)
