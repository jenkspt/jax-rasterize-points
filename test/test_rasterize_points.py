import numpy as np
import jax.numpy as jnp
from jax.experimental import enable_x64

from rasterize_points import *


def test_floatbits2int():
    z = jnp.array([1, -1, 0, 2, -2], dtype=jnp.float32)
    # test identity
    assert all(intbits2float(floatbits2int(z)) == z)
    # sorted ints should be same order as sorted floats
    assert all(intbits2float(jnp.sort(floatbits2int(z))) == jnp.sort(z))


def test_pack_fragments():
    z = jnp.array([1, -1, 0, 2, -2], dtype=jnp.float32)
    i = jnp.arange(len(z), dtype=jnp.int32)
    zi = pack_fragments(z)
    z2, i2 = unpack_fragments(zi)
    # Identity test
    assert all(z == z2)
    assert all(i == i2)
    with enable_x64():
        assert all(unpack_fragments(jnp.sort(zi))[0] == jnp.sort(z))