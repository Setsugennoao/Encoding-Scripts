import lvsfunc as lvf


def sb(s: str):
  return sum(bytearray(s, 'utf-8'))


grain_amount = sb("Cat") * sb("Sugar") / sb("Desert") / 10 ** 3
mitchell = lvf.kernels.Mitchell()

degrain_args = [(500, 750), (1000, 500), (1400, 700)]

knl_args = [
    dict(h=2, a=4, d=1, s=2),
    dict(h=1.35, a=2, d=1, s=4),
    dict(h=1.35, a=2, d=1, s=4),
]
