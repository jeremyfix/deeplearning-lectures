#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np


pt = np.linspace(0, 1)

plt.figure()
for gamma in [0, 1, 5]:
    # plt.plot(pt, -((1 - pt) ** gamma) * np.log(pt), label=f"$\\gamma = {gamma}$")
    plt.plot(pt, (1 - pt) ** gamma, label=f"$\\gamma = {gamma}$")
plt.legend()
plt.show()
