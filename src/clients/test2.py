

import numpy as np
import matplotlib.pyplot as plt
import torch


ncwh = torch.rand((2, 1, 100, 100))
for i in range(ncwh.shape[0]):
    plt.imshow(ncwh[i, 0, :, :], cmap='gray')
    plt.show()