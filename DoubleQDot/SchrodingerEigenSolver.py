from Utils import ScalarField, EigenState, EigenStates

import numpy as np
from scipy.sparse.linalg import eigsh

def solve(potential: ScalarField, stateAmounts: int, fourierResolution: int) -> EigenStates:

    # Check for potential dim and grid width/height, warn if they're not good composite numbers.
    if (potential.ndim != 2):
        raise Exception("The potential field given is not 2D!")

    gridWidth   = potential.shape[0]
    gridHeight  = potential.shape[1]

    # Estimate eigensolver execution time.

    # FFT: potential
    potential_FT = np.fft.fft2(potential)

    # Construct Hamiltonian in k-space
    fR2 = fourierResolution * fourierResolution
    hamiltonian_FT = np.zeros(shape = (fR2, fR2), dtype = complex)

    for ix in range(fourierResolution):
        for iy in range(fourierResolution):
            for jx in range(fourierResolution):
                for jy in range(fourierResolution):
                    i = ix + iy * fourierResolution
                    j = jx + jy * fourierResolution

                    if (i == j):
                        hamiltonian_FT[i][j] += (np.square(ix / gridWidth) + np.square(iy / gridHeight)) * np.square(2.0 * np.pi)

                    kx = ky = 0

                    if (jx >= ix):
                        kx = jx - ix
                    else:
                        kx = jx - ix + gridWidth

                    if (jy >= iy):
                        ky = jy - iy
                    else:
                        ky = jy - iy + gridHeight
                        
                    hamiltonian_FT[i][j] += potential_FT[kx][ky]

    # According to symmetry, the imaginary part of hamiltonian_FT is omitted
    eigenValues, eigenVectors = eigsh(hamiltonian_FT.real, stateAmounts)

    eigenVectors = np.transpose(eigenVectors)

    eigenStates = []

    for val, vec in zip(eigenValues, eigenVectors):
        waveFunction_FT_FirstQuadrant = vec.reshape((fourierResolution, fourierResolution))
        waveFunction_FT = np.zeros(shape = (gridWidth, gridHeight))

        # Fill the 4 quadrants
        for i in range(fourierResolution):
            for j in range(fourierResolution):
                waveFunction_FT[i                           ][j                             ] = waveFunction_FT_FirstQuadrant[i][j]
                waveFunction_FT[(gridWidth - i) % gridWidth ][j                             ] = waveFunction_FT_FirstQuadrant[i][j]
                waveFunction_FT[i                           ][(gridHeight - j) % gridHeight ] = waveFunction_FT_FirstQuadrant[i][j]
                waveFunction_FT[(gridWidth - i) % gridWidth ][(gridHeight - j) % gridHeight ] = waveFunction_FT_FirstQuadrant[i][j]

        waveFunction = np.fft.ifft2(waveFunction_FT)
        eigenValue = val / (gridHeight * gridWidth)

        eigenStates.append(EigenState(waveFunction, eigenValue))

    return eigenStates