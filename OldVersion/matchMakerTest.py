import numpy as np
import matplotlib.pyplot as plt
import random as rand

maxRadius = 4
def generatePoint(radius):
    angle = rand.random()*2*np.pi
    x = maxRadius + radius*np.cos(angle)
    y = maxRadius + radius*np.sin(angle)
    x = round(x)
    y = round(y)
    return [x, y]


samples = np.ceil(np.random.exponential(scale=1.0, size=10000))
dots = np.array([generatePoint(point) for point in samples])
plt.scatter(dots[:,0],dots[:,1])
plt.show()