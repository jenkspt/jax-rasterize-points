from typing import Tuple, NamedTuple
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp

#from jax.experimental import enable_x64
jax.config.update('jax_enable_x64', True)


@partial(jax.jit, static_argnums=1)
def floatbits2int(f: jnp.ndarray, positive=False) -> jnp.ndarray:
    assert f.dtype == jnp.float32
    i = f.view(jnp.int32)
    if positive: return i
    return jnp.where(i < 0, -(jnp.int32(0x80000000) + i), i)


@partial(jax.jit, static_argnums=1)
def intbits2float(i: jnp.ndarray, positive=False) -> jnp.ndarray:
    assert i.dtype == jnp.int32
    if positive: return i.view(jnp.float32)
    i = jnp.where(i < 0, -(jnp.int32(0x80000000) + i), i)
    return i.view(jnp.float32)


@jax.jit
def pack_fragments(depth: jnp.ndarray, index: jnp.ndarray=None) -> jnp.ndarray:
    assert depth.dtype == jnp.float32
    shape = depth.shape
    if index is None:
        # we want an index over the last dimension of depth
        index = jnp.arange(shape[-1], dtype=jnp.int32).reshape(*(1,) * (depth.ndim-1), -1)
    else:
        assert depth.shape == index.shape
    # int32 in last axis will be converted to single int64
    packed = jnp.empty((*shape, 2), dtype=jnp.int32)
    packed = packed.at[..., 0].set(index)
    packed = packed.at[..., 1].set(floatbits2int(depth, True))
    return packed.view(jnp.int64).squeeze(-1)


@jax.jit
def unpack_fragments(fragments: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    assert fragments.dtype == jnp.int64
    shape = fragments.shape
    fragments = fragments.reshape(*shape, 1).view(jnp.int32)
    index = fragments[..., 0]
    depth = intbits2float(fragments[..., 1], True)
    return depth, index


@jax.jit
def ndc_to_screen(ndc: jnp.ndarray, image_size: Tuple[int, int]) -> jnp.ndarray:
    return (ndc + 1) / 2 * (jnp.array(image_size) - 1) + .5


class Fragments(NamedTuple):
    zbuf: jnp.array
    idxs: jnp.array

    @property
    def valid(self):
        return self.idxs < jnp.iinfo(jnp.int32).max


# Sentinal value is float32(inf), int32(-1) packed into a single int64
SENTINAL_FRAGMENT = pack_fragments(jnp.float32(jnp.inf), jnp.int32(jnp.iinfo(jnp.int32).max))


@jax.jit
def get_pixels(xy, z, eps=1e-10):
    pixels = (xy - .5).round(0).astype(jnp.int32)
    m = jnp.any(pixels < 0, axis=-1) | (z < eps)
    return jnp.where(m[..., None], jnp.iinfo(jnp.int32).max, pixels)


@partial(jax.jit, static_argnums=(1,2))
def rasterize_points_min_depth(
        points: jnp.ndarray,
        image_size: Tuple[int, int],
        ndc=True) -> Fragments:
    """
    Args:
        image_size:
        points:             # [..., N, 3]
    """
    assert points.shape[-1] == 3
    assert len(image_size) == 2
    raster_packed = jnp.full((*points.shape[:-2], *image_size), SENTINAL_FRAGMENT, dtype=jnp.int64)
    
    xy, depth = points[..., :2], points[..., 2]
    if ndc:
        xy = ndc_to_screen(xy, image_size)
    # Round xy points to nearest pixel value
    pixels = get_pixels(xy, depth)
    x, y = jnp.rollaxis(pixels, -1)
    # here we pack depth and index values into a single int64
    zi = pack_fragments(depth)
    # Keep track of the minimum z value and its index
    raster_packed = raster_packed.at[..., y, x].min(zi, mode='drop')
    # Unpack int64 into corresponding depth and index
    depth, index = unpack_fragments(raster_packed)
    return Fragments(depth, index)


def _rasterize_points_iter(points, image_size, k=1, ndc=True):
    xy, depth = points[..., :2], points[..., 2]

    if ndc:
        xy = ndc_to_screen(xy, image_size)
    # Negative indices will wrap in jax/numpy so need to set to positive out-of-bounds value
    # Do the same for negative depth
    pixels = get_pixels(xy, depth)
    # packed depth and index
    zi = pack_fragments(depth)

    raster_packed = jnp.full((*points.shape[:-2], *image_size), SENTINAL_FRAGMENT, dtype=jnp.int64)

    for i in range(k):
        x, y = jnp.rollaxis(pixels, -1)
        # here we pack depth and index values into a single int64
        # Keep track of the minimum depth w/ index
        raster_packed = raster_packed.at[..., y, x].min(zi, mode='drop')
        # Unpack int64 into corresponding depth and index
        zbuf, index = unpack_fragments(raster_packed)
        yield Fragments(zbuf, index)

        if i < (k - 1):
            # peele already rasterized points (set them to out-of-bounds so they aren't used again)
            pixels = pixels.at[..., index, :].set(np.iinfo(jnp.int32).max, mode='drop')
            # reset raster to sentinal values
            raster_packed = raster_packed.at[...].set(SENTINAL_FRAGMENT)


@partial(jax.jit, static_argnums=(1,2,3))
def rasterize_points(points, image_size, k=1, ndc=True):
    assert k >= 1
    zbuf = jnp.empty((*points.shape[:-2], *image_size, k), dtype=jnp.float32)
    idxs = jnp.empty((*points.shape[:-2], *image_size, k), dtype=jnp.int32)

    for i, ri in enumerate(_rasterize_points_iter(points, image_size, k, ndc)):
        zbuf, idxs = zbuf.at[..., i].set(ri.zbuf), idxs.at[..., i].set(ri.idxs)
    return Fragments(zbuf, idxs)


@partial(jax.jit, static_argnums=(3,4,5,6))
def render_points(points, colors, background_color, image_size, k=4, blend=1, ndc=True):
    assert k > 0
    xy = points[..., :2]
    if ndc:
        xy = ndc_to_screen(xy, image_size)

    px_center = (xy - .5).round(0) + .5
    _alphas = (1 - jnp.sqrt(jnp.sum((px_center - xy)**2, -1)))**blend
    transmit = 1
    image, depth_image = 0, 0

    for ri in _rasterize_points_iter(points, image_size, k, ndc):
        alphas = _alphas.at[..., ri.idxs].get(mode='fill', fill_value=0)
        w = alphas * transmit
        image = image + w[..., None] * colors.at[..., ri.idxs, :].get(mode='fill', fill_value=0)
        depth_image = depth_image + w * points.at[..., ri.idxs].get(mode='fill', fill_value=jnp.float32(jnp.inf))
        transmit = transmit * (1 - alphas)

    image = image + transmit * background_color
    return image, depth_image


def depth2disp(depth, eps=1e-10):
    return 1 / jnp.maximum(depth, eps)


""" Default indexer"""
def _di(a, i, d):
    return a.at[i].get(mode='fill', fill_value=d)