import math
import statistics
import random
import os
import glob

print(math.pi)
print(math.cos(2*math.pi))

liste = [1, 4, 6, 2, 5]

print(statistics.mean(liste))
print(statistics.variance(liste))

random.seed(0)
print(random.choice(liste))
print(random.random())
print(random.randint(5, 10))

random.sample(range(100), 10)

print('liste de départ', liste)

random.shuffle(liste)
print('liste mélangée', liste)

os.getcwd()

print(glob.glob('*'))