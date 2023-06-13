import numpy as np
import pycsou.util.ptype as pyct
from hvox import _nufft

__all__ = ["vis2dirty", "dirty2vis", "compute_psf"]


def vis2dirty(
    uvw_l: pyct.NDArray,
    xyz: pyct.NDArray,
    vis: pyct.NDArray,
    real: bool = True,
    epsilon: float = 1e-4,
    wgt_vis: pyct.NDArray = None,
    wgt_dirty: pyct.NDArray = None,
    w_term: bool = True,
    normalisation: str = "xyz",
    **kwargs,
) -> pyct.NDArray:

    r"""
    Converts visibilities to a dirty image using the HVOX algorithm [HVOXpaper]_.

    This implementation relies on the finufft [FINUFFT]_ and Pycsou [Pycsou]_ packages.
    It supports a "chunked" strategy to subdivide the operation into smaller tasks or chunks to
    optimize memory allocation while keeping competitive performance.

    Parameters
    ----------
    uvw_l: NDArray[real]
        (N_vis, 3) normalized UVW coordinates.
    xyz: NDArray[real]
        (N_pix, 3) sky pixels in \bS^{2}.
    vis: NDArray[complex]
        (..., N_vis) visibilities to transform.
    real: bool = True
        Output dirty images are real-valued.
    epsilon: float = 1e-4
        Requested relative accuracy >= 0.
        If epsilon=0, the transform is computed exactly via direct evaluation of the exponential
        sum using a Numba JIT-compiled kernel.
    wgt_vis: NDArray[real, complex] = None
        (N_vis,) weights to apply to visibilities before transforming.
    wgt_dirty: NDArray[real, complex] = None
        (N_pix,) weights to apply to pixels after transforming.
    w_term: bool = True
        If false, drop the W-term entirely. (2D type-3 transform.)
    normalisation: str["xyz", "uvw_l", "both"]
        Divide the output pixel values by:
            * "xyz": the total number of pixels.
            * "uvw_l": the sum of `wgt_vis` if defined, else by the number of visibilities.
            * "both": both.

    kwargs: dict
        Extra kwargs passed to
        :py:class:`~pycsou.operator.linop.fft.nufft.NUFFT`.

        Supported parameters for :py:func:`pycsou.operator.linop.fft.nufft.NUFFT.type3` are:

            * enable_warnings: bool = True
            * chunked: bool = True
            * parallel: bool = True

        Supported parameters for `finufft.Plan <https://finufft.readthedocs.io/en/latest/python.html#finufft.Plan>`_.

            * nthreads: int = None
                Default to N_core / 2

        Default values are chosen if unspecified.

    Returns
    -------
    dirty: NDArray[real, complex]
        (..., N_pix) computed dirty images.

    Notes
    -----
    The dtype of the input visibilities `vis` determines the precision at which computations are done.
    `finufft.Plan <https://finufft.readthedocs.io/en/latest/python.html#finufft.Plan>`_ n_trans will be set based on
    number leading terms in `vis`.

    Example
    -------

    .. plot::

       import matplotlib.pyplot as plt
       import numpy as np
       import hvox

       rng = np.random.default_rng(0)
       n_vis = 40
       uvw_lambda = np.zeros((n_vis, 3))
       uvw_lambda[:n_vis//2] = rng.random((n_vis // 2, 3)) - 0.5
       uvw_lambda[n_vis//2:] = -uvw_lambda[:n_vis//2]
       x = np.linspace(0, 1, 25)
       xx, yy = np.meshgrid(x, x)
       zz = rng.random(xx.shape) - 0.5
       xyz = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], axis=1)
       visibilities = rng.random(n_vis).astype("complex")
       visibilities += 1j * rng.random(n_vis)

       dirty = hvox.vis2dirty(
           uvw_l=uvw_lambda,
           xyz=xyz,
           vis=visibilities,
       ).reshape(25, 25)
       plt.imshow(dirty)
       plt.xlabel("x")
       plt.ylabel("y")
       plt.title("Dirty image")

    See Also
    --------
    :py:func:`~hvox.core.dirty2vis`, :py:func:`~hvox.core.compute_psf`

    """

    # Remove w_term if asked
    uvw_ = uvw_l if w_term else uvw_l[:, :-1]
    xyz_ = xyz if w_term else xyz[:, :-1]

    # Apply visibility weights (flagged weights
    vis = vis * wgt_vis if wgt_vis is not None else vis

    # NUFFT Type 3
    dirty = _nufft.nufft_vis2dirty(xyz=xyz_, uvw_lambda=uvw_, visibilities=vis, real=real, epsilon=epsilon,
                                   nufft_kwargs=kwargs)

    # Apply dirty weights
    dirty = dirty * wgt_dirty if wgt_dirty is not None else dirty

    if normalisation is not None:
        if normalisation in ["uvw", "both"]:
            dirty /= wgt_vis.sum() if wgt_vis is not None else len(uvw_l)
        if normalisation in ["xyz", "both"]:
            dirty /= len(xyz)

    return dirty


def dirty2vis(
    uvw_l: pyct.NDArray,
    xyz: pyct.NDArray,
    dirty: pyct.NDArray,
    epsilon: float = 1e-4,
    wgt_dirty: pyct.NDArray = None,
    wgt_vis: pyct.NDArray = None,
    w_term: bool = True,
    normalisation="xyz",
    **kwargs,
) -> pyct.NDArray:
    r"""
    Converts a dirty image to visibilities using the HVOX algorithm [HVOXpaper]_.

    This implementation relies on the finufft [FINUFFT]_ and Pycsou [Pycsou]_ packages.
    It supports a "chunked" strategy to subdivide the operation into smaller tasks or chunks to
    optimize memory allocation while keeping competitive performance.

    Parameters
    ----------
    uvw_l: NDArray[real]
        (N_vis, 3) normalized UVW coordinates.
    xyz: NDArray[real]
        (N_pix, 3) sky pixels in \bS^{2}.
    dirty: NDArray[real, complex]
        (..., N_pix) dirty images to transform.
    epsilon: float = 1e-4
        Requested relative accuracy >= 0.
        If epsilon=0, the transform is computed exactly via direct evaluation of the exponential
        sum using a Numba JIT-compiled kernel.
    wgt_dirty: NDArray[real, complex] = None
        (N_pix,) weights to apply to pixels before transforming.
    wgt_vis: NDArray[real, complex] = None
        (N_vis,) weights to apply to visibilities after transforming.
    w_term: bool = True
        If false, drop the W-term entirely. (2D type-3 transform.)
    normalisation: str["xyz", "uvw_l", "both"]
        Divide the output pixel values by:
            * "xyz": the total number of pixels.
            * "uvw_l": the sum of `wgt_vis` if defined, else by the number of visibilities.
            * "both": both.
    kwargs: dict
        Extra kwargs passed to
        :py:class:`~pycsou.operator.linop.fft.nufft.NUFFT`.

        Supported parameters for :py:func:`pycsou.operator.linop.fft.nufft.NUFFT.type3` are:

            * enable_warnings: bool = True
            * chunked: bool = True
            * parallel: bool = True

        Supported parameters for `finufft.Plan <https://finufft.readthedocs.io/en/latest/python.html#finufft.Plan>`_.

            * nthreads: int = None
                Default to N_core / 2

        Default values are chosen if unspecified.

    Returns
    -------
    vis: NDArray[complex]
        (..., N_vis) computed visibilities.

    Notes
    -----
    The dtype of the input dirty image `dirty` determines the precision at which computations are done.
    `finufft.Plan <https://finufft.readthedocs.io/en/latest/python.html#finufft.Plan>`_ n_trans will be set based on
    number leading terms in `dirty`.

    Example
    -------

    .. plot::

       import matplotlib.pyplot as plt
       import numpy as np
       import hvox

       rng = np.random.default_rng(0)
       n_vis = 40
       uvw_lambda = np.zeros((n_vis, 3))
       uvw_lambda[:n_vis // 2] = rng.random((n_vis // 2, 3)) - 0.5
       uvw_lambda[n_vis // 2:] = -uvw_lambda[:n_vis // 2]
       x = np.linspace(0, 1, 25)
       xx, yy = np.meshgrid(x, x)
       zz = rng.random(xx.shape) - 0.5
       xyz = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], axis=1)
       dirty = rng.random(xx.size)

       visibilities = hvox.dirty2vis(
           uvw_l=uvw_lambda,
           xyz=xyz,
           dirty=dirty,
       )
       uvw_dist = np.linalg.norm(uvw_lambda, axis=1)
       plt.scatter(uvw_dist, visibilities.real, label="real part")
       plt.scatter(uvw_dist, visibilities.imag, label="imaginary part")
       plt.xlabel("uvw_dist")
       plt.ylabel("visibility")
       plt.legend()

    See Also
    --------
    :py:func:`~hvox.core.vis2dirty`, :py:func:`~hvox.core.compute_psf`

    """

    nrows, _ = uvw_l.shape

    # Remove w_term if asked
    uvw_ = uvw_l if w_term else uvw_l[:, :-1]
    xyz_ = xyz if w_term else xyz[:, :-1]

    # Apply dirty weights
    dirty = dirty * wgt_dirty if wgt_dirty is not None else dirty

    # Get NUFFT args
    nufft_kwargs = dict(
        enable_warnings=kwargs.get("enable_warnings", True),
        chunked=kwargs.get("chunked", True),
        parallel=kwargs.get("parallel", True),
        max_mem=kwargs.get("max_mem", 512),
        nthreads=kwargs.get("nthreads", None),
    )
    # NUFFT Type 3
    vis = _nufft.nufft_dirty2vis(xyz=xyz_,
                                 uvw_lambda=uvw_,
                                 dirty=dirty,
                                 real=True,
                                 epsilon=epsilon,
                                 nufft_kwargs=nufft_kwargs)

    # Apply visibility weights (flagged weights)
    vis = vis * wgt_vis if wgt_vis is not None else vis

    if normalisation is not None:
        if normalisation in ["uvw", "both"]:
            vis /= wgt_vis.sum() if wgt_vis is not None else len(uvw_l)
        if normalisation in ["xyz", "both"]:
            vis /= len(xyz)

    return vis


def compute_psf(
    uvw_l: pyct.NDArray,
    xyz: pyct.NDArray,
    xyz_center: pyct.NDArray = None,
    epsilon: float = 1e-4,
    wgt_vis: pyct.NDArray = None,
    wgt_dirty: pyct.NDArray = None,
    w_term: bool = True,
    normalisation: str = "xyz",
    **kwargs,
) -> pyct.NDArray:
    r"""
    Computes the point-spread function (PSF) using the HVOX algorithm [HVOXpaper]_.

    This implementation relies on the finufft [FINUFFT]_ and Pycsou [Pycsou]_ packages.
    It supports a "chunked" strategy to subdivide the operation into smaller tasks or chunks to
    optimize memory allocation while keeping competitive performance.

    Parameters
    ----------
    uvw_l: NDArray[real]
        (N_vis, 3) normalized UVW coordinates.
    xyz: NDArray[real]
        (N_pix, 3) sky pixels in \bS^{2}.
    xyz_center: NDArray[real]
        (..., 3) PSF center locations in the sky.
    epsilon: float = 1e-4
        Requested relative accuracy >= 0.
        If epsilon=0, the transform is computed exactly via direct evaluation of the exponential
        sum using a Numba JIT-compiled kernel.
    wgt_vis: NDArray[real, complex] = None
        (N_vis,) weights to apply to visibilities before transforming.
    wgt_dirty: NDArray[real, complex] = None
        (N_pix,) weights to apply to pixels after transforming.
    w_term: bool = True
        If false, drop the W-term entirely. (2D type-3 transform.)
    normalisation: str["xyz", "uvw_l", "both"]
        Divide the output pixel values by:
            * "xyz": the total number of pixels.
            * "uvw_l": the sum of `wgt_vis` if defined, else by the number of visibilities.
            * "both": both.

    kwargs: dict
        Extra kwargs passed to
        :py:class:`~pycsou.operator.linop.fft.nufft.NUFFT`.

        Supported parameters for :py:func:`pycsou.operator.linop.fft.nufft.NUFFT.type3` are:

            * enable_warnings: bool = True
            * chunked: bool = True
            * parallel: bool = True

        Supported parameters for `finufft.Plan <https://finufft.readthedocs.io/en/latest/python.html#finufft.Plan>`_.

            * nthreads: int = None
                Default to N_core / 2

        Default values are chosen if unspecified.

    Returns
    -------
    psf: NDArray[real]
        (..., N_pix) computed psf.

    Notes
    -----
    The dtype of the input visibilities `xyz_center` determines the precision at which computations are done.
    `finufft.Plan <https://finufft.readthedocs.io/en/latest/python.html#finufft.Plan>`_ n_trans will be set based on
    number leading terms in `xyz_center`.

    Example
    -------

    .. plot::

       import matplotlib.pyplot as plt
       import numpy as np
       import hvox

       rng = np.random.default_rng(0)
       n_vis = 40
       uvw_lambda = np.zeros((n_vis, 3))
       uvw_lambda[:n_vis // 2] = rng.random((n_vis // 2, 3)) - 0.5
       uvw_lambda[n_vis // 2:] = -uvw_lambda[:n_vis // 2]
       x = np.linspace(0, 1, 25)
       xx, yy = np.meshgrid(x, x)
       zz = rng.random(xx.shape) - 0.5
       xyz = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], axis=1)

       psf = hvox.compute_psf(
           uvw_l=uvw_lambda,
           xyz=xyz,
       ).reshape(25, 25)
       plt.imshow(psf)
       plt.xlabel("x")
       plt.ylabel("y")
       plt.title("PSF")

    See Also
    --------
    :py:func:`~hvox.core.dirty2vis`, :py:func:`~hvox.core.dirty2vis`

    """

    dtype = np.csingle if np.issubdtype(uvw_l.dtype, np.single) else np.cdouble
    if xyz_center is None:
        xyz_center = np.zeros_like(uvw_l[0])

    phase_center = np.exp(-1j * 2 * np.pi * (uvw_l.dot(xyz_center))).astype(dtype)

    return vis2dirty(uvw_l=uvw_l,
                     xyz=xyz,
                     vis=phase_center,
                     wgt_vis=wgt_vis,
                     wgt_dirty=wgt_dirty,
                     normalisation=normalisation,
                     w_term=w_term,
                     epsilon=epsilon,
                     kwargs=kwargs)

