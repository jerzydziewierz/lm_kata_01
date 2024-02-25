import unittest
from ndnumbers.cplx import Cplx
import jax.numpy as jnp


class TestCplx(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)  # add assertion here

    def test_cplx_create(self):
        # new : the preferred way to create is from a N*2 array
        c = Cplx(jnp.array([[1, 2], [3, 4]]))  # first dimension is the batch dimension; the second dimension is the element-of-object dimension
        self.assertEqual(c.e0[0], 1)
        self.assertEqual(c.e0[1], 3)
        self.assertEqual(c.e1[0], 2)
        self.assertEqual(c.e1[1], 4)
        self.assertEqual(c.norm[0], 2.236068)
        self.assertEqual(c.norm[1], 5.000000)

    def test_mul_cplx_cplx(self):
        a = Cplx(jnp.array([[1, 2], [3, 4]]))
        b = Cplx(jnp.array([[5, 6], [7, 8]]))
        c = a * b  # this should be an element-by-element multiplication
        self.assertEqual(c.e0[0], -7)
        self.assertEqual(c.e1[0], 16)
        self.assertEqual(c.e0[1], -11)
        self.assertEqual(c.e1[1], 52)
        self.assertAlmostEqual(c.norm[0], 17.464, 2)
        self.assertAlmostEqual(c.norm[1], 53.150, 2)


if __name__ == '__main__':
    unittest.main()
