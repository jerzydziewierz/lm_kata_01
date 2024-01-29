import jax.numpy as jnp
import jax


def raw_mul(a, b):
    a0 = a[:, 0]
    a1 = a[:, 1]
    a2 = a[:, 2]
    a3 = a[:, 3]
    a4 = a[:, 4]
    a5 = a[:, 5]
    a6 = a[:, 6]
    a7 = a[:, 7]
    b0 = b[:, 0]
    b1 = b[:, 1]
    b2 = b[:, 2]
    b3 = b[:, 3]
    b4 = b[:, 4]
    b5 = b[:, 5]
    b6 = b[:, 6]
    b7 = b[:, 7]
    # multiplication actual, as per https://www.euclideanspace.com/maths/algebra/realNormedAlgebra/octonion/index.htm
    # I bet this could be expressed as a set of matrix ops, but for now, hand coding as if it was cuda:
    # c1 is everywhere that results in "e1" or "-e1"
    # e.t.c.
    # implementation idea: scatter the b-vector to a matrix, and then do a vector*matrix mul.
    c0 = a0 * b0 + -a1 * b1 + -a2 * b2 + -a3 * b3 + -a4 * b4 + -a5 * b5 + -a6 * b6 + -a7 * b7
    c1 = a0 * b1 + a1 * b0 + a2 * b4 + a3 * b7 + -a4 * b2 + a5 * b6 + -a6 * b5 + -a7 * b3
    c2 = a0 * b2 + -a1 * b4 + a2 * b0 + a3 * b5 + a4 * b1 + -a5 * b3 + a6 * b7 + -a7 * b6
    c3 = a0 * b3 + -a1 * b7 + -a2 * b5 + -a3 * b0 + a4 * b6 + a5 * b2 + -a6 * b4 + a7 * b1
    c4 = a0 * b4 + a1 * b2 + -a2 * b1 + -a3 * b6 + a4 * b0 + a5 * b7 + a6 * b3 + -a7 * b5
    c5 = a0 * b5 + -a1 * b6 + a2 * b3 + -a3 * b2 + -a4 * b7 + a5 * b0 + a6 * b1 + a7 * b4
    c6 = a0 * b6 + a1 * b5 + -a2 * b7 + a3 * b4 + -a4 * b3 + a5 * b1 + a6 * b0 + a7 * b2
    c7 = a0 * b7 + a1 * b3 + a2 * b6 + -a3 * b1 + a4 * b5 + -a5 * b4 + -a6 * b2 + a7 * b0
    combined = jnp.stack([c0, c1, c2, c3, c4, c5, c6, c7], axis=-1)
    return combined


def octo_mul(a, b):
    if isinstance(a, jnp.ndarray):
        a = Octonion(a)
    if isinstance(b, jnp.ndarray):
        b = Octonion(b)
    if isinstance(a, Octonion) and isinstance(b, Octonion):
        return Octonion(raw_mul(a.x, b.x))
    elif isinstance(a, Octonion) and isinstance(b, float):
        return Octonion(a.x * b)
    elif isinstance(a, float) and isinstance(b, Octonion):
        return Octonion(a * b.x)
    elif isinstance(a, Octonion) and isinstance(b, int):
        return Octonion(a.x * b)
    elif isinstance(a, int) and isinstance(b, Octonion):
        return Octonion(a * b.x)
    elif isinstance(a, Octonion) and isinstance(b, jnp.ndarray):
        return Octonion(raw_mul(a.x, b))
    elif isinstance(a, jnp.ndarray) and isinstance(b, Octonion):
        return Octonion(raw_mul(a, b.x))
    else:
        raise ValueError(f'type(a)={type(a)}, type(b)={type(b)}')


class Octonion:
    def __init__(self, x: jnp.array):
        if isinstance(x, Octonion):
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
        if isinstance(self, Octonion):
            if isinstance(other, Octonion):
                return Octonion(self.x + other.x)
            else:
                if isinstance(other, float):
                    # my bet is that this will be sufficient to produce "bias", in the sense that, there is no need to have an octonion bias; the octonian can alaways get rotated and scaled in the first phase of swiglu so that the bias gets applied as needed.
                    return Octonion(self.x + jnp.broadcast_to(jnp.array(other), self.x.shape))
        raise ValueError(f'type(other)={type(other)}')

    def __mul__(self, other):
        return octo_mul(self, other)

    def __repr__(self):
        return f"Octonions, shape={self.x.shape}, first one is ({self.x[0, 0]:0.1f}, {self.x[0, 1]:0.1f}i, {self.x[0, 2]:0.1f}j, {self.x[0, 3]:0.1f}k, {self.x[0, 4]:0.1f}l, {self.x[0, 5]:0.1f}li, {self.x[0, 6]:0.1f}lj, {self.x[0, 7]:0.1f}lk)"


raw_null = jnp.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.float32).reshape(1, 8)
raw_unit = jnp.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.float32).reshape(1, 8)
raw_i = jnp.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=jnp.float32).reshape(1, 8)
raw_j = jnp.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=jnp.float32).reshape(1, 8)
raw_k = jnp.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=jnp.float32).reshape(1, 8)
raw_l = jnp.array([0, 0, 0, 0, 1, 0, 0, 0], dtype=jnp.float32).reshape(1, 8)
raw_li = jnp.array([0, 0, 0, 0, 0, 1, 0, 0], dtype=jnp.float32).reshape(1, 8)
raw_lj = jnp.array([0, 0, 0, 0, 0, 0, 1, 0], dtype=jnp.float32).reshape(1, 8)
raw_lk = jnp.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=jnp.float32).reshape(1, 8)

null = Octonion(raw_null)
unit = Octonion(raw_unit)
i = Octonion(raw_i)
j = Octonion(raw_j)
k = Octonion(raw_k)
l = Octonion(raw_l)
li = Octonion(raw_li)
lj = Octonion(raw_lj)
lk = Octonion(raw_lk)
