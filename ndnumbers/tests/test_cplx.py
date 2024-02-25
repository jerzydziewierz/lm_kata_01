import unittest
from ndnumbers.cplx import Cplx
import jax.numpy as jnp


class Test_Cplx(unittest.TestCase):

    def test_cplx_create(self):
        # new : the preferred way to create is from a N*2 array
        c = Cplx(jnp.array([[1, 2], [3,
                                     4]]))  # first dimension is the batch dimension; the second dimension is the element-of-object dimension
        self.assertEqual(c.e0[0], 1)
        self.assertEqual(c.e0[1], 3)
        self.assertEqual(c.e1[0], 2)
        self.assertEqual(c.e1[1], 4)
        self.assertEqual(c.norm[0], 2.236068)
        self.assertEqual(c.norm[1], 5.000000)

    def test_mul_cplx_cplx(self):
        """
        element-wise multiplication.
        :return:
        """
        a = Cplx(jnp.array([[1, 2], [3, 4]]))
        b = Cplx(jnp.array([[5, 6], [7, 8]]))
        c = a * b  # this should be an element-by-element multiplication
        self.assertEqual(c.e0[0], -7)
        self.assertEqual(c.e1[0], 16)
        self.assertEqual(c.e0[1], -11)
        self.assertEqual(c.e1[1], 52)
        self.assertAlmostEqual(c.norm[0], 17.464, 2)
        self.assertAlmostEqual(c.norm[1], 53.150, 2)

    def test_add_cplx_cplx(self):
        """
        element-wise add.
        :return:
        """
        a = Cplx(jnp.array([[1, 0], [0, 1], [1, 1]]))
        b = Cplx(jnp.array([[0, 1], [0, 0], [1, 1]]))
        c = a + b
        self.assertEqual(c.e0[0], 1)
        self.assertEqual(c.e1[0], 1)
        self.assertEqual(c.e0[1], 0)
        self.assertEqual(c.e1[1], 1)
        self.assertEqual(c.e0[2], 2)
        self.assertEqual(c.e1[2], 2)
        self.assertAlmostEqual(c.norm[0], 1.414, 2)
        self.assertAlmostEqual(c.norm[1], 1.000, 2)
        self.assertAlmostEqual(c.norm[2], 2.828, 2)

    def test_add_cplx_scalar_broadcast(self):
        """
        Element-wise add of complex and a scalar.
        The scalar is added to the real part only, and the complex part is pass-through.
        Adding a non-matching size or type gives a helpful error message.
        Adding a single scalar to cplx is a broadcast.

        """
        a = Cplx(jnp.array([[1, 0], [0, 1], [1, 1]]))
        b_single = 1

        c_single = a + b_single

        self.assertEqual(c_single.e0[0], 2)
        self.assertEqual(c_single.e1[0], 0)
        self.assertEqual(c_single.e0[1], 1)
        self.assertEqual(c_single.e1[1], 1)
        self.assertEqual(c_single.e0[2], 2)
        self.assertEqual(c_single.e1[2], 1)

    def test_add_cplx_scalar_elementwise(self):
        """
        Element-wise add of complex and a scalar.
        Adding a matched size array to cplx is an element-wise add (like, bias)
        adding a non-matching size or type gives a helpful error message.
        """
        a = Cplx(jnp.array([[1, 0], [0, 1], [1, 1]]))
        b_array = jnp.array([1, 2, 3])
        c_array = a + b_array
        self.assertEqual(c_array.e0[0], 2)
        self.assertEqual(c_array.e1[0], 0)
        self.assertEqual(c_array.e0[1], 2)
        self.assertEqual(c_array.e1[1], 1)
        self.assertEqual(c_array.e0[2], 4)
        self.assertEqual(c_array.e1[2], 1)

    def test_cplx_relu0(self):
        """
        element-wise leaky relu with a fixed alpha.
        relu0 processes real part only, and leaves complex part pass-through.
        see relu1 for a version that processes both real and imaginary parts.
        :return:
        """
        pass

    def test_cplx_relu1(self):
        """
        element-wise leaky relu with a fixed alpha.
        relu1 processes both real and imaginary parts.
        :return:
        """
        pass


if __name__ == '__main__':
    unittest.main()
