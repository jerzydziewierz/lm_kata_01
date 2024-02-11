import jax.numpy as jnp
import jax


def raw_mul(a, b):
    a0 = a[:, 0]
    a1 = a[:, 1]
    b0 = b[:, 0]
    b1 = b[:, 1]
    b2 = b[:, 2]
    c0 = a0 * b0 - a1 * b1
    c1 = a0 * b1 + a1 * b0
    combined = jnp.stack([c0, c1], axis=-1)
    return combined


def cplx_mul(a, b) -> Cplx:
    if isinstance(a, jnp.ndarray):
        a = Cplx(a)
    if isinstance(b, jnp.ndarray):
        b = Cplx(b)
    if isinstance(a, Cplx) and isinstance(b, Cplx):
        return Cplx(raw_mul(a.x, b.x))
    elif isinstance(a, Cplx) and isinstance(b, float):
        return Cplx(a.x * b)
    elif isinstance(a, float) and isinstance(b, Cplx):
        return Cplx(a * b.x)
    elif isinstance(a, Cplx) and isinstance(b, int):
        return Cplx(a.x * b)
    elif isinstance(a, int) and isinstance(b, Cplx):
        return Cplx(a * b.x)
    elif isinstance(a, Cplx) and isinstance(b, jnp.ndarray):
        return Cplx(raw_mul(a.x, b))
    elif isinstance(a, jnp.ndarray) and isinstance(b, Cplx):
        return Cplx(raw_mul(a, b.x))
    else:
        raise ValueError(f'type(a)={type(a)}, type(b)={type(b)}')


class Cplx:
    def __init__(self, x: jnp.array):
        if isinstance(x, Cplx):
            self.x = x.x
        else:
            # assert the last dimension is 8:
            assert x.shape[-1] == 2
            if len(x.shape) == 1:
                x = x.reshape(1, 2)
            self.x = x
        self.norm = self.call_norm()

    def to_jnp(self):
        return self.x

    def call_norm(self):
        return jnp.linalg.norm(self.x, axis=-1)

    def __add__(self, other):
        if isinstance(self, Cplx):
            if isinstance(other, Cplx):
                return Cplx(self.x + other.x)
            else:
                if isinstance(other, float):
                    # my bet is that this will be sufficient to produce "bias", in the sense that, there is no need to have an octonion bias; the octonian can alaways get rotated and scaled in the first phase of swiglu so that the bias gets applied as needed.
                    return Octonion(self.x + jnp.broadcast_to(jnp.array(other), self.x.shape))
        raise ValueError(f'type(other)={type(other)}')

    def __mul__(self, other):
        return cplx_mul(self, other)

    def __repr__(self):
        return f"Complex, shape={self.x.shape}, first one is ({self.x[0, 0]:0.1f}, i{self.x[0, 1]:0.1f})"


raw_null = jnp.array([0, 0], dtype=jnp.float32).reshape(1, 2)
raw_unit = jnp.array([1, 0], dtype=jnp.float32).reshape(1, 2)
raw_i = jnp.array([0, 1], dtype=jnp.float32).reshape(1, 2)

null = Cplx(raw_null)
unit = Cplx(raw_unit)
i = Cplx(raw_i)
