from Utils import ScalarField

def singleQDotPotential(depth: float, wellWidth: int, totalWidth: int) -> ScalarField:
    
    field = ScalarField(shape = (totalWidth, totalWidth), dtype = complex)

    for i in range(totalWidth):
        for j in range(totalWidth):
            x = i - totalWidth / 2
            y = j - totalWidth / 2

            rr = x * x + y * y

            if rr < wellWidth * wellWidth / 4:
                field[i][j] = depth * (rr / (wellWidth * wellWidth / 4) - 1.0)
            else:
                field[i][j] = 0.0

    return field

def doubleQDotPotential(gridNum: int, gridSize: float) -> ScalarField:
    
    ######################## WIP

    field = np.ndarray(shape = (2 * gridNum, gridNum), dtype = complex)

    return field
