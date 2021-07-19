import lvsfunc as lvf


def sb(s: str):
  return sum(bytearray(s, 'utf-8'))


grain_amount = sb("Cat") * sb("Sugar") / sb("Desert") / 10 ** 3
mitchell = lvf.kernels.Mitchell()
