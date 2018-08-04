# Some types for abstraction

from typing import List
from dataclasses import dataclass

import numpy as np

ScalarField = np.ndarray
        
@dataclass
class EigenState:
    waveFunction: ScalarField
    energy: float

EigenStates = List[EigenState]

# Plotting

from matplotlib import pyplot as plt 

def plot(field: ScalarField):
    plt.imshow(field, interpolation='bilinear')
    plt.colorbar()
    plt.show()

    