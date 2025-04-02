import random

c = {
    'test': lambda: random.sample(['type1'] * 4 + ['type2'] * 3 + ['type3'] * 2, k=9)
}

b = c['test']()[0]
print(b)