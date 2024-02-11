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
    a8 = a[:, 8]
    a9 = a[:, 9]
    aA = a[:, 10]
    aB = a[:, 11]
    aC = a[:, 12]
    aD = a[:, 13]
    aE = a[:, 14]
    aF = a[:, 15]

    b0 = b[:, 0]
    b1 = b[:, 1]
    b2 = b[:, 2]
    b3 = b[:, 3]
    b4 = b[:, 4]
    b5 = b[:, 5]
    b6 = b[:, 6]
    b7 = b[:, 7]
    b8 = b[:, 8]
    b9 = b[:, 9]
    bA = b[:, 10]
    bB = b[:, 11]
    bC = b[:, 12]
    bD = b[:, 13]
    bE = b[:, 14]
    bF = b[:, 15]
    # multiplication actual, as per https://www.euclideanspace.com/maths/algebra/realNormedAlgebra/octonion/index.htm
    # I bet this could be expressed as a set of matrix ops, but for now, hand coding as if it was cuda:
    # c1 is everywhere that results in "e1" or "-e1"
    # e.t.c.
    # implementation idea: scatter the b-vector to a matrix, and then do a vector*matrix mul.
    c0 = jnp.nan
    c1 = jnp.nan
    c2 = jnp.nan
    c3 = jnp.nan
    c4 = jnp.nan
    c5 = jnp.nan
    c6 = jnp.nan
    c7 = jnp.nan
    c8 = jnp.nan
    c9 = jnp.nan
    cA = jnp.nan
    cB = jnp.nan
    cC = jnp.nan
    cD = jnp.nan
    cE = jnp.nan
    cF = jnp.nan
    combined = jnp.stack([c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, cA, cB, cC, cD, cE, cF], axis=-1)
    return combined


def sept_mul(a, b):
    if isinstance(a, jnp.ndarray):
        a = Sedenion(a)
    if isinstance(b, jnp.ndarray):
        b = Sedenion(b)
    if isinstance(a, Sedenion) and isinstance(b, Sedenion):
        return Sedenion(raw_mul(a.x, b.x))
    elif isinstance(a, Sedenion) and isinstance(b, float):
        return Sedenion(a.x * b)
    elif isinstance(a, float) and isinstance(b, Sedenion):
        return Sedenion(a * b.x)
    elif isinstance(a, Sedenion) and isinstance(b, int):
        return Sedenion(a.x * b)
    elif isinstance(a, int) and isinstance(b, Sedenion):
        return Sedenion(a * b.x)
    elif isinstance(a, Sedenion) and isinstance(b, jnp.ndarray):
        return Sedenion(raw_mul(a.x, b))
    elif isinstance(a, jnp.ndarray) and isinstance(b, Sedenion):
        return Sedenion(raw_mul(a, b.x))
    else:
        raise ValueError(f'type(a)={type(a)}, type(b)={type(b)}')


class Sedenion:
    def __init__(self, x: jnp.array):
        if isinstance(x, Sedenion):
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
        if isinstance(self, Sedenion):
            if isinstance(other, Sedenion):
                return Sedenion(self.x + other.x)
            else:
                if isinstance(other, float):
                    # my bet is that this will be sufficient to produce "bias", in the sense that, there is no need to have an octonion bias; the octonian can alaways get rotated and scaled in the first phase of swiglu so that the bias gets applied as needed.
                    return Sedenion(self.x + jnp.broadcast_to(jnp.array(other), self.x.shape))
        raise ValueError(f'type(other)={type(other)}')

    def __mul__(self, other):
        return sept_mul(self, other)

    def __repr__(self):
        return f"Sedenion, shape={self.x.shape}, first one is ({self.x[0, 0]:0.1f}, i{self.x[0, 1]:0.1f}, j{self.x[0, 2]:0.1f}j, k{self.x[0, 3]:0.1f}, {self.x[0, 4]:0.1f}l, {self.x[0, 5]:0.1f}li, {self.x[0, 6]:0.1f}lj, {self.x[0, 7]:0.1f}lk)"


raw_null = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.float32).reshape(1, 16)
raw_e0 = jnp.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.float32).reshape(1, 16)
raw_e1 = jnp.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.float32).reshape(1, 16)
raw_e2 = jnp.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.float32).reshape(1, 16)
raw_e3 = jnp.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.float32).reshape(1, 16)
raw_e4 = jnp.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.float32).reshape(1, 16)
raw_e5 = jnp.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.float32).reshape(1, 16)
raw_e6 = jnp.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.float32).reshape(1, 16)
raw_e7 = jnp.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.float32).reshape(1, 16)
raw_e8 = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.float32).reshape(1, 16)
raw_e9 = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=jnp.float32).reshape(1, 16)
raw_eA = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=jnp.float32).reshape(1, 16)
raw_eB = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=jnp.float32).reshape(1, 16)
raw_eC = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=jnp.float32).reshape(1, 16)
raw_eD = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=jnp.float32).reshape(1, 16)
raw_eE = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=jnp.float32).reshape(1, 16)
raw_eF = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=jnp.float32).reshape(1, 16)

raw_unit = raw_e0

null = Sedenion(raw_null)
unit = Sedenion(raw_unit)
e0 = Sedenion(raw_e0)
e1 = Sedenion(raw_e1)
e2 = Sedenion(raw_e2)
e3 = Sedenion(raw_e3)
e4 = Sedenion(raw_e4)
e5 = Sedenion(raw_e5)
e6 = Sedenion(raw_e6)
e7 = Sedenion(raw_e7)
e8 = Sedenion(raw_e8)
e9 = Sedenion(raw_e9)
eA = Sedenion(raw_eA)
eB = Sedenion(raw_eB)
eC = Sedenion(raw_eC)
eD = Sedenion(raw_eD)
eE = Sedenion(raw_eE)
eF = Sedenion(raw_eF)

