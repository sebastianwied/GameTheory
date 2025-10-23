import numpy as np

data = np.array([[1,2,3,4], [2,3,4,5], [3,4,5,6], [4,5,6,7]])
x = np.array([np.mean(mean) for mean in data])
print(x)