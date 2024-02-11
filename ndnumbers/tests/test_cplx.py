import unittest
from multid.cplx import Cplx


class TestCplx(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_cplx_create(self):
        c = Cplx(jnp.array([1, 2]))
        self.assertEqual(c.x[0], 1)
        self.assertEqual(c.x[1], 2)
        self.assertEqual(c.norm, 2.236068)
        c = Cplx(jnp.array([[1, 2], [3, 4]]))
        self.assertEqual(c.x[0, 0], 1)
        self.assertEqual(c.x[0, 1], 2)
        self.assertEqual(c.x[1, 0], 3)
        self.assertEqual(c.x[1, 1], 4)
        self.assertEqual(c.norm[0], 2.236068)
        self.assertEqual(c.norm[1], 5.000000)


if __name__ == '__main__':
    unittest.main()
