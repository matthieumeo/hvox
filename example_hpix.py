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
    import astropy_healpix
    from astropy_healpix import HEALPix
    import healpy as hp

    from astropy.coordinates import Galactic
    from astropy.io import fits
except:
    raise ValueError("Missing packages")

import hvox

if __name__ == "__main__":
    ra = 15.0 * u.deg
    dec = -45.0 * u.deg
    rad = 5.0 * u.deg
    print("Allocating visibilities / Creating Low-SKA configuration")
    lowr3 = create_named_configuration("LOWBD2", rmax=6_00.0)
    vis = create_visibility(
        config=lowr3,
        times=np.linspace(-4, 4, 9) * np.pi / 12,
        frequency=np.r_[15e7],
        channel_bandwidth=np.r_[5e4],
        weight=1.0,
        phasecentre=SkyCoord(
            ra=ra , dec=dec, frame="icrs", equinox="J2000"
        ),
        polarisation_frame=PolarisationFrame("stokesI"),
        times_are_ha=False,
    )

    print("Loading sky GT")
    hvox_path = os.sep.join(os.path.abspath(hvox.__file__).split(os.sep)[:-3])
    fits_path = os.path.join(hvox_path, "data", "models", "haslam408_dsds_Remazeilles2014.fits")

    # Load fits
    hdulist = fits.open(fits_path)
    nside = hdulist[1].header['NSIDE']
    order = hdulist[1].header['ORDERING']
    # Create healpix object
    haslam_hpix = HEALPix(nside=nside, order=order, frame=Galactic())
    # get cone of interest
    haslam_coi = haslam_hpix.cone_search_lonlat(ra, dec, radius=rad)
    # get map values
    haslam_full = hp.read_map(fits_path)
    # get values of interest
    haslam = haslam_full[haslam_coi]
    # get ra and dec (lat and lon) coordinaes
    directions = haslam_hpix.healpix_to_skycoord(haslam_coi)
    # convert o lmn
    lmn = skycoord_to_lmn(directions, SkyCoord(ra=ra, dec=dec, frame="icrs", equinox="J2000"))
    lmn = np.asarray(lmn).T

    xyz = lmn
    # dont use jacobian for healpix
    wgt_dirty = None # 1 / jacobian

    print("Simulating visibilities")
    vis_estimate = hvox.dirty2vis(
        uvw_lambda=vis.visibility_acc.uvw_lambda.reshape(-1, 3),
        xyz=xyz.reshape(-1, 3),
        dirty=haslam.reshape(-1),
        wgt_dirty=wgt_dirty,
        chunked=True,
    )

    print("Estimating dirty image")
    sky_estimate = (
        hvox.vis2dirty(
            uvw_lambda=vis.visibility_acc.uvw_lambda.reshape(-1, 3),
            xyz=xyz.reshape(-1, 3),
            vis=vis_estimate.reshape(-1),
            wgt_vis=vis.visibility_acc.flagged_weight.reshape(-1),
            wgt_dirty=wgt_dirty,
            chunked=True,
        )
    )
    print("Estimating PSF")
    psf_estimate = (
        hvox.compute_psf(
            uvw_lambda=vis.visibility_acc.uvw_lambda.reshape(-1, 3),
            xyz=xyz.reshape(-1, 3),
            wgt_vis=vis.visibility_acc.flagged_weight.reshape(-1),
            wgt_psf=wgt_dirty,
            chunked=True,
        )
    )
    fig, ax = plt.subplots(
        figsize=(4, 4),
        subplot_kw={"projection": "3d"},
    )
    ids = np.random.permutation(len(xyz))[:1000]
    ax.scatter(*xyz[ids].T, alpha=0.1)
    plt.show()

    haslam_plot = hp.ma(haslam_full)
    haslam_plot.mask = np.ones_like(haslam_full, dtype=bool)
    haslam_plot.mask[haslam_coi] = False


    dirty_plot = haslam_plot.copy()
    dirty_plot[haslam_coi] = sky_estimate

    psf_plot = haslam_plot.copy()
    psf_plot[haslam_coi] = psf_estimate

    hp.mollview(
        abs(haslam_plot),
        norm="log",
    )
    hp.graticule()
    plt.show()

    hp.mollview(
        abs(dirty_plot),
        norm="log",
    )
    hp.graticule()
    plt.show()

    hp.mollview(
        abs(psf_plot),
        norm="log",
    )
    hp.graticule()
    plt.show()
    print("Done!")
