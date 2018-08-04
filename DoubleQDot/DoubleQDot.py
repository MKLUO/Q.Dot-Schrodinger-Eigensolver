import sys

from SchrodingerEigenSolver import solve
from Potential import singleQDotPotential
from Utils import plot

import numpy as np

def main():

    SQDpotential  = singleQDotPotential(depth = 2.0, wellWidth = 40, totalWidth = 100).real
    
    eigenStates = solve(potential = SQDpotential, stateAmounts = 10, fourierResolution = 10)

    for eigenState in eigenStates:
        plot(eigenState.waveFunction.real**2 + eigenState.waveFunction.imag**2)
        plot(np.angle(eigenState.waveFunction))
    

if __name__ == "__main__":
    sys.exit(int(main() or 0))

