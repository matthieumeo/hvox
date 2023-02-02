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

    import healpy as hp
except:
    raise ValueError("Missing packages")

import hvox

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
    hvox_path = os.sep.join(os.path.abspath(hvox.__file__).split(os.sep)[:-3])
    fits_path = os.path.join(hvox_path, "data", "models", "haslam408_dsds_Remazeilles2014.fits")
    haslam, header = hp.read_map(fits_path, h=True)

    direction_cosines, jacobian = get_direction_cosines(haslam)
    wgt_dirty = 1 / jacobian.reshape(-1)

    print("Simulating visibilities")
    vis_estimate = hvox.dirty2vis(
        uvw_lambda=vis.visibility_acc.uvw_lambda.reshape(-1, 3),
        xyz=direction_cosines.reshape(-1, 3),
        dirty=m31image.pixels.data.reshape(-1),
        wgt_dirty=wgt_dirty,
        chunked=True,
    )

    print("Estimating dirty image")
    sky_estimate = (
        hvox.vis2dirty(
            uvw_lambda=vis.visibility_acc.uvw_lambda.reshape(-1, 3),
            xyz=direction_cosines.reshape(-1, 3),
            vis=vis_estimate.reshape(-1),
            wgt_vis=vis.visibility_acc.flagged_weight.reshape(-1),
            wgt_dirty=wgt_dirty,
            chunked=True,
        )
        .reshape(m31image.pixels.shape)
        .squeeze()
    )
    print("Estimating PSF")
    psf_estimate = (
        hvox.compute_psf(
            uvw_lambda=vis.visibility_acc.uvw_lambda.reshape(-1, 3),
            xyz=direction_cosines.reshape(-1, 3),
            wgt_vis=vis.visibility_acc.flagged_weight.reshape(-1),
            wgt_dirty=wgt_dirty,
            chunked=True,
        )
        .reshape(m31image.pixels.shape)
        .squeeze()
    )

    hp.mollview(
        haslam,
        norm="log",
    )
    hp.graticule()
    plt.show()

    print("Done!")
