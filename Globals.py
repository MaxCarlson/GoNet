
BoardLength = 19
BoardSize = BoardLength ** 2
InputPlanes = 5
BoardDepth = 1 + InputPlanes * 2

# Padded board sizes to make things easier to compute
BoardLengthP = BoardLength + 2
BoardSizeP = BoardLengthP ** 2

EMPTY     = 0
BLACK     = 1
WHITE     = 2
OFF_BOARD = 3