import matplotlib.pyplot as plt
import numpy as np
from util import getScatters


p1 = np.array([3.0, 20.0])
p2 = np.array([6.0, -20.0])

res1, res2 = getScatters(p1, p2, 10, 1)
print(res1, res2)
plt.scatter(p1[0], p1[1])
plt.scatter(p2[0], p2[1])
plt.scatter(res1[:,0], res1[:,1])
plt.scatter(res2[:,0], res2[:,1])
plt.show()

print("123")
