from numpy import array, array_equal, flipud
from src.transform import Transform, Rotate90, Flip


def test_transform():
    b = array([[1,  1, -1],
               [-1, 1,  1],
               [1, -1, -1]])
    t = Transform(Rotate90(2), Flip(flipud))
    transformed_b = t.transform(b)
    assert array_equal(transformed_b, array([[-1,  1,  1],
                                             [1,   1, -1],
                                             [-1, -1,  1]]))
    reversed_b = t.reverse(transformed_b)
    assert array_equal(reversed_b,  b)
