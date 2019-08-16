import numpy as np
import numpy.random
import matplotlib.pyplot as plt

depth = 10000

def f(x, y):
  return x/(1-x)+(1-y)

xmax = ymax = int(depth / 10)
z = numpy.array([[f(x/depth, y/depth) for x in range(1, xmax)] for y in range(1, ymax)])

d = [x/depth for x in range(1, xmax)]
p = [y/depth for y in range(1, ymax)]

plt.pcolormesh(z)
plt.colorbar()
curves = 10
m = max([max(row) for row in z])
levels = numpy.arange(0, m, (1 / float(curves)) * m)
plt.contour(z, colors="white", levels=levels)
plt.xticks([])
plt.yticks([])
plt.xlabel("d (0.0001 - 0.1)")
plt.ylabel("p (0.0001 - 0.1)")


plt.show()