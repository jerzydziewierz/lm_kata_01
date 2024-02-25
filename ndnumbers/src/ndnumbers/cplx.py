import jax.tree_util as tu
import jax.numpy as jnp
import jax


def raw_mul(a, b):
    a0 = a[:, 0]
    a1 = a[:, 1]

    b0 = b[:, 0]
    b1 = b[:, 1]

    c0 = a0 * b0 - a1 * b1
    c1 = a0 * b1 + a1 * b0
    combined = jnp.stack([c0, c1], axis=-1)
    return combined


def cplx_mul(a, b):
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


# pre-declare the class "Cplx" so that I can use it as a result type:
class Cplx:
    pass


def mul_cplx_cplx(a, b):
    c_e0 = a.e0 * b.e0 - a.e1 * b.e1
    c_e1 = a.e0 * b.e1 + a.e1 * b.e0
    return Cplx(c_e0, c_e1)


@tu.register_pytree_node_class
class Cplx:  # noqa: F811
    def __init__(self, e0=None, e1=None):
        # copy constructor
        if isinstance(e0, Cplx):
            self.e0 = e0.e0
            self.e1 = e0.e1
            return

        # construct from (*,2) array
        if isinstance(e0, jnp.ndarray) and e1 is None and e0.shape[-1] == 2:
            self.e0 = e0[:, 0]
            self.e1 = e0[:, 1]
            return

        # construct from two arguments
        if isinstance(e0, jnp.ndarray) and isinstance(e1, jnp.ndarray) and e0.shape == e1.shape:
            self.e0 = e0
            self.e1 = e1
            return
        # construct from two arguments but do not test for type. This might be the jax tracer
        # print('constructing anyway...')
        if e0 is not None and e1 is not None:
            self.e0 = e0
            self.e1 = e1
            return

        raise ValueError(f'unknown constructor for argument with e0={e0} and e1={e1}')

    def to_jnp(self):
        return jnp.stack([self.e0, self.e1], axis=-1)

    @property
    def norm(self):
        return jnp.linalg.norm(self.to_jnp(), axis=-1)

    def __add__(self, other):
        if isinstance(self, Cplx):
            if isinstance(other, Cplx):
                return Cplx(self.e0 + other.e0, self.e1 + other.e1)
            elif isinstance(other, float) or isinstance(other, int):
                """
                my bet is that it will be sufficient to use "scalar bias", 
                in the sense that, there is no need to have an octonion bias; 
                the octonion can always get rotated and scaled in the first phase of swiglu, 
                so that the bias gets applied as needed.
                """
                return Cplx(self.e0 + jnp.broadcast_to(jnp.array(other), self.e0.shape), self.e1)
            elif isinstance(other, jnp.ndarray):
                """verify the shape of `other` and if not, give a helpful error message"""
                if other.shape == self.e0.shape:
                    return Cplx(self.e0 + other, self.e1)
                else:
                    raise ValueError(f"shape mismatch: self.e0.shape={self.e0.shape}, other.shape={other.shape}")

        raise ValueError(f'type(other)={type(other)}')

    def __mul__(self, other):
        if isinstance(other, Cplx):
            return mul_cplx_cplx(self, other)
        if isinstance(other, float):
            return Cplx(self.e0 * other, self.e1 * other)
        if isinstance(other, jnp.ndarray):
            # scale each element of the array
            if other.shape == self.e0.shape:
                return Cplx(self.e0 * other, self.e1 * other)
        raise ValueError(f"I do not know how to multiply me by {type(other)}")

    def __rmul__(self, other):
        # multplication may not be commutative. Preserve order:
        if isinstance(other, Cplx):
            return mul_cplx_cplx(other, self)
        if isinstance(other, float) or isinstance(other, int):
            return Cplx(self.e0 * other, self.e1 * other)
        if isinstance(other, jnp.ndarray):
            # scale each element of the array
            if other.shape == self.e0.shape:
                return Cplx(self.e0 * other, self.e1 * other)
        raise ValueError(f"I do not know how to multiply {type(other)} by myself")

    def __neg__(self):
        return cplx_mul(self, -1)

    def __abs__(self):
        return self.call_norm()

    def __repr__(self):
        return self.s

    @property
    def s(self):
        """
        short version of the string descriptor
        :return:
        """
        # if the size of e1 is ():
        if isinstance(self.e0, int):
            return f"({self.e0:+1d}, i{self.e1:+1d})"
        if isinstance(self.e0, float):
            return f"({self.e0:+0.1f}, i{self.e1:+0.1f})"
        if hasattr(self.e0, 'shape'):
            if self.e0.shape == ():
                return f"({self.e0:+0.1f}, i{self.e1:+0.1f})"
            if self.e0.shape[-1] == 1:
                return f"({self.e0[0]:+0.1f}, i{self.e1[0]:+0.1f})"
            else:
                return f"cplx.shape={self.e0.shape}"
        else:
            return "cplx?"

    @property
    def shape(self):
        return self.e0.shape

    @property
    def shortstring(self):
        return self.s

    def __getitem__(self, key):
        return Cplx(self.e0[key], self.e1[key])

    def tree_flatten(node):
        children = (node.e0, node.e1)
        aux_data = ()
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # print("tree_unflatten:",aux_data, children)
        return cls(children[0], children[1])


raw_null = jnp.array([0, 0], dtype=jnp.float32).reshape(1, 2)
raw_unit = jnp.array([1, 0], dtype=jnp.float32).reshape(1, 2)
raw_i = jnp.array([0, 1], dtype=jnp.float32).reshape(1, 2)

null = Cplx(raw_null)
unit = Cplx(raw_unit)
i = Cplx(raw_i)
