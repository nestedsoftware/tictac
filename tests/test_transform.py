import numpy as np
from tictac.transform import Transform, Rotate90, Flip


def test_transform():
    b = np.array([[1,  1, -1],
                  [-1, 1,  1],
                  [1, -1, -1]])
    t = Transform(Rotate90(2), Flip(np.flipud))

    transformed_b = t.transform(b)

    assert np.array_equal(transformed_b, np.array([[-1,  1,  1],
                                                   [1,   1, -1],
                                                   [-1, -1,  1]]))
    reversed_b = t.reverse(transformed_b)

    assert np.array_equal(reversed_b,  b)
