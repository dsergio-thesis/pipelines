

import numpy as np
import matplotlib.pyplot as plt

# plot arcsin(x)
x = np.linspace(-10, 10, 100)
y = np.arcsin(x)
plt.plot(x, y)
plt.title('Arcsin Function')
plt.xlabel('x')
plt.ylabel('arcsin(x)')
plt.grid()
plt.show()
