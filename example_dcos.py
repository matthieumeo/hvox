import os

import matplotlib.pyplot as plt
import numpy as np

try:
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from rascil.processing_components import create_test_image, show_image
    from ska_sdp_datamodels.configuration.config_create import (
        create_named_configuration,
    )
    from ska_sdp_datamodels.science_data_model.polarisation_model import (
        PolarisationFrame,
    )
    from ska_sdp_datamodels.visibility import create_visibility
    from ska_sdp_func_python.util.coordinate_support import skycoord_to_lmn
except:
    raise ValueError("Missing packages. To run example install the additional dependencies (see repository README)")

import hvox

hvox_path = os.sep.join(os.path.abspath(hvox.__file__).split(os.sep)[:-3])
os.environ["RASCIL_DATA"] = os.path.join(hvox_path, "data")


def get_direction_cosines(image):
    _, _, _, npixel = image.pixels.data.shape
    _, _, ny, nx = image["pixels"].shape
    lmesh, mmesh = np.meshgrid(np.arange(ny), np.arange(nx))
    ra_grid, dec_grid = image.image_acc.wcs.sub([1, 2]).wcs_pix2world(lmesh, mmesh, 0)
    ra_grid = np.deg2rad(ra_grid)
    dec_grid = np.deg2rad(dec_grid)
    directions = SkyCoord(
        ra=ra_grid.ravel() * u.rad,
        dec=dec_grid.ravel() * u.rad,
        frame="icrs",
        equinox="J2000",
    )
    l, m, _ = skycoord_to_lmn(directions, image.image_acc.phasecentre)
    jacobian = np.sqrt(1 - l**2 - m**2)
    direction_cosines = np.stack([l, m, jacobian - 1.0], axis=-1).reshape(-1, 3)
    return direction_cosines, jacobian


if __name__ == "__main__":
    print("Allocating visibilities / Creating Low-SKA configuration")
    lowr3 = create_named_configuration("LOWBD2", rmax=3_000.0)
    vis = create_visibility(
        config=lowr3,
        times=np.linspace(-4, 4, 9) * np.pi / 12,
        frequency=np.r_[15e7],
        channel_bandwidth=np.r_[5e4],
        weight=1.0,
        phasecentre=SkyCoord(
            ra=+15.0 * u.deg, dec=-45.0 * u.deg, frame="icrs", equinox="J2000"
        ),
        polarisation_frame=PolarisationFrame("stokesI"),
        times_are_ha=False,
    )

    print("Loading sky GT")
    m31image = create_test_image(
        phasecentre=vis.phasecentre, frequency=np.r_[15e7], cellsize=5e-4
    )
    direction_cosines, jacobian = get_direction_cosines(m31image)
    wgt_dirty = 1 / jacobian.reshape(-1)

    print("Simulating visibilities")
    vis_estimate = hvox.dirty2vis(uvw_l=vis.visibility_acc.uvw_lambda.reshape(-1, 3),
                                  xyz=direction_cosines.reshape(-1, 3),
                                  dirty=m31image.pixels.data.reshape(-1),
                                  wgt_vis=vis.visibility_acc.flagged_weight.reshape(-1),
                                  wgt_dirty=wgt_dirty,
                                  normalisation="xyz",
                                  chunked=True)

    print("Estimating dirty image")
    dirty_estimate = (
        hvox.vis2dirty(uvw_l=vis.visibility_acc.uvw_lambda.reshape(-1, 3),
                       xyz=direction_cosines.reshape(-1, 3),
                       vis=vis_estimate.reshape(-1),
                       wgt_vis=vis.visibility_acc.flagged_weight.reshape(-1),
                       wgt_dirty=wgt_dirty,
                       normalisation="xyz",
                       chunked=True)
        .reshape(m31image.pixels.shape)
        .squeeze()
    )
    print("Estimating PSF")
    psf_estimate = (
        hvox.compute_psf(
            uvw_l=vis.visibility_acc.uvw_lambda.reshape(-1, 3),
            xyz=direction_cosines.reshape(-1, 3),
            wgt_vis=vis.visibility_acc.flagged_weight.reshape(-1),
            wgt_dirty=wgt_dirty,
            normalisation="both",
            chunked=False,
        )
        .reshape(m31image.pixels.shape)
        .squeeze()
    )

    print("Showing results")
    fig, axs = plt.subplots(
        1,
        3,
        figsize=(15, 4),
        subplot_kw={"projection": m31image.image_acc.wcs.sub([1, 2])},
    )
    im1 = axs[0].imshow(
        m31image.pixels.data.squeeze(), origin="lower", cmap="cubehelix"
    )
    axs[0].set_xlabel(m31image.image_acc.wcs.wcs.ctype[0])
    axs[0].set_ylabel(m31image.image_acc.wcs.wcs.ctype[1])
    axs[0].set_title("Sky (ground truth)")
    plt.colorbar(im1, ax=axs[0])
    im2 = axs[1].imshow(psf_estimate, origin="lower", cmap="cubehelix")
    axs[1].set_xlabel(m31image.image_acc.wcs.wcs.ctype[0])
    axs[1].set_ylabel(m31image.image_acc.wcs.wcs.ctype[1])
    axs[1].set_title("Estimated PSF")
    plt.colorbar(im2, ax=axs[1])
    im3 = axs[2].imshow(dirty_estimate, origin="lower", cmap="cubehelix")
    axs[2].set_xlabel(m31image.image_acc.wcs.wcs.ctype[0])
    axs[2].set_ylabel(m31image.image_acc.wcs.wcs.ctype[1])
    axs[2].set_title("Dirty")
    plt.colorbar(im3, ax=axs[2])
    plt.show()

    print("Done!")

    from ska_sdp_func_python.imaging import invert_ng, predict_ng

    vis_ng = vis.copy(deep=True)
    vis_ng = predict_ng(vis_ng, m31image)
    vis_estimate = hvox.dirty2vis(uvw_l=vis.visibility_acc.uvw_lambda.reshape(-1, 3),
                                  xyz=direction_cosines.reshape(-1, 3), dirty=m31image.pixels.data.reshape(-1), normalisation=None, chunked=True)

    plt.scatter(vis_ng.vis.data.ravel(), vis_estimate.ravel())
    plt.xlabel("NG"), plt.ylabel("HVOX")
    plt.show()

    dirty_ng = m31image.copy(deep=True)
    dirty_ng = invert_ng(vis_ng, dirty_ng, dopsf=False, normalise=True)
    psf_ng = m31image.copy(deep=True)
    psf_ng = invert_ng(vis_ng, psf_ng, dopsf=True, normalise=True)
    dirty_estimate = (
        hvox.vis2dirty(uvw_l=vis.visibility_acc.uvw_lambda.reshape(-1, 3), xyz=direction_cosines.reshape(-1, 3),
                       vis=vis_estimate.reshape(-1), wgt_vis=vis.visibility_acc.flagged_weight.reshape(-1),
                       normalisation="uvw", chunked=True)
        .reshape(m31image.pixels.shape)
        .squeeze()
    )

    psf_estimate = (
        hvox.compute_psf(
            uvw_l=vis.visibility_acc.uvw_lambda.reshape(-1, 3),
            xyz=direction_cosines.reshape(-1, 3),
            wgt_vis=vis.visibility_acc.flagged_weight.reshape(-1),
            wgt_dirty=wgt_dirty,
            normalisation="uvw",
            chunked=True,
        )
        .reshape(m31image.pixels.shape)
        .squeeze()
    )

    fig, axs = plt.subplots(
        2,
        2,
        figsize=(10, 10),
        subplot_kw={"projection": m31image.image_acc.wcs.sub([1, 2])},
    )
    im1 = axs[0, 0].imshow(
        dirty_ng[0].pixels.data.squeeze(), origin="lower", cmap="cubehelix"
    )
    axs[0, 0].set_xlabel(m31image.image_acc.wcs.wcs.ctype[0])
    axs[0, 0].set_ylabel(m31image.image_acc.wcs.wcs.ctype[1])
    axs[0, 0].set_title("DIRTY NG")
    plt.colorbar(im1, ax=axs[0, 0])
    im2 = axs[0, 1].imshow(
        psf_ng[0].pixels.data.squeeze(), origin="lower", cmap="cubehelix"
    )
    axs[0, 1].set_xlabel(m31image.image_acc.wcs.wcs.ctype[0])
    axs[0, 1].set_ylabel(m31image.image_acc.wcs.wcs.ctype[1])
    axs[0, 1].set_title("PSF NG")
    plt.colorbar(im2, ax=axs[0, 1])

    im1 = axs[1, 0].imshow(dirty_estimate, origin="lower", cmap="cubehelix")
    axs[1, 0].set_xlabel(m31image.image_acc.wcs.wcs.ctype[0])
    axs[1, 0].set_ylabel(m31image.image_acc.wcs.wcs.ctype[1])
    axs[1, 0].set_title("DIRTY HVOX")
    plt.colorbar(im1, ax=axs[1, 0])
    im2 = axs[1, 1].imshow(psf_estimate, origin="lower", cmap="cubehelix")
    axs[1, 1].set_xlabel(m31image.image_acc.wcs.wcs.ctype[0])
    axs[1, 1].set_ylabel(m31image.image_acc.wcs.wcs.ctype[1])
    axs[1, 1].set_title("PSF HVOX")
    plt.colorbar(im2, ax=axs[1, 1])
    plt.show()
