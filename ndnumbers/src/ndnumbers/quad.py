import jax.numpy as jnp
import jax


def raw_mul(a, b):
    a0 = a[:, 0]
    a1 = a[:, 1]
    a2 = a[:, 2]
    a3 = a[:, 3]
    b0 = b[:, 0]
    b1 = b[:, 1]
    b2 = b[:, 2]
    b3 = b[:, 3]
    # multiplication actual, as per https://www.euclideanspace.com/maths/algebra/realNormedAlgebra/octonion/index.htm
    # I bet this could be expressed as a set of matrix ops, but for now, hand coding as if it was cuda:
    # c1 is everywhere that results in "e1" or "-e1"
    # e.t.c.
    # implementation idea: scatter the b-vector to a matrix, and then do a vector*matrix mul.
    c0 = a0 * b0 + -a1 * b1 + -a2 * b2 + -a3 * b3
    c1 = a0 * b1 + +a1 * b0 + +a2 * b3 + -a3 * b2
    c2 = a0 * b2 + -a1 * b3 + +a2 * b0 + +a3 * b1
    c3 = a0 * b3 + +a1 * b2 + -a2 * b1 + +a3 * b0
    combined = jnp.stack([c0, c1, c2, c3], axis=-1)
    return combined


def quad_mul(a, b) -> Quaternion:
    if isinstance(a, jnp.ndarray):
        a = Quaternion(a)
    if isinstance(b, jnp.ndarray):
        b = Quaternion(b)
    if isinstance(a, Quaternion) and isinstance(b, Quaternion):
        return Quaternion(raw_mul(a.x, b.x))
    elif isinstance(a, Quaternion) and isinstance(b, float):
        return Quaternion(a.x * b)
    elif isinstance(a, float) and isinstance(b, Quaternion):
        return Quaternion(a * b.x)
    elif isinstance(a, Quaternion) and isinstance(b, int):
        return Quaternion(a.x * b)
    elif isinstance(a, int) and isinstance(b, Quaternion):
        return Quaternion(a * b.x)
    elif isinstance(a, Quaternion) and isinstance(b, jnp.ndarray):
        return Quaternion(raw_mul(a.x, b))
    elif isinstance(a, jnp.ndarray) and isinstance(b, Quaternion):
        return Quaternion(raw_mul(a, b.x))
    else:
        raise ValueError(f'type(a)={type(a)}, type(b)={type(b)}')


class Quaternion:
    def __init__(self, x: jnp.array):
        if isinstance(x, Quaternion):
            self.x = x.x
        else:
            # assert the last dimension is 8:
            assert x.shape[-1] == 8
            if len(x.shape) == 1:
                x = x.reshape(1, 8)
            self.x = x
        self.norm = self.call_norm()

    def to_jnp(self):
        return self.x

    def call_norm(self):
        return jnp.linalg.norm(self.x, axis=-1)

    def __add__(self, other):
        if isinstance(self, Quaternion):
            if isinstance(other, Quaternion):
                return Quaternion(self.x + other.x)
            else:
                if isinstance(other, float):
                    return Quaternion(self.x + jnp.broadcast_to(jnp.array(other), self.x.shape))
        raise ValueError(f'type(other)={type(other)}')

    def __mul__(self, other):
        return quad_mul(self, other)

    def __repr__(self):
        return f"Quaternions, shape={self.x.shape}, first one is ({self.x[0, 0]:0.1f}, i{self.x[0, 1]:0.1f}, j{self.x[0, 2]:0.1f}, k{self.x[0, 3]:0.1f}k)"


raw_null = jnp.array([0, 0, 0, 0], dtype=jnp.float32).reshape(1, 4)
raw_unit = jnp.array([1, 0, 0, 0], dtype=jnp.float32).reshape(1, 4)
raw_i = jnp.array([0, 1, 0, 0], dtype=jnp.float32).reshape(1, 4)
raw_j = jnp.array([0, 0, 1, 0], dtype=jnp.float32).reshape(1, 4)
raw_k = jnp.array([0, 0, 0, 1], dtype=jnp.float32).reshape(1, 4)

null = Quaternion(raw_null)
unit = Quaternion(raw_unit)
i = Quaternion(raw_i)
j = Quaternion(raw_j)
k = Quaternion(raw_k)

