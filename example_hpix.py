import os

import matplotlib.pyplot as plt
import numpy as np

try:
    import astropy.units as u
    import astropy_healpix
    import healpy as hp
    from astropy.coordinates import Galactic, SkyCoord
    from astropy.io import fits
    from astropy_healpix import HEALPix
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
    raise ValueError("Missing packages")

import hvox

if __name__ == "__main__":

    # Sparse or dense sky
    sparse_sky = False

    # Radius of FOV
    radius = 20.0 * u.deg

    print("Creating Low-SKA configuration")
    # Chose same coordinate frame for visibilities and healpix map
    phasecentre = SkyCoord(frame="galactic", b=5.0 * u.deg, l=-15.0 * u.deg)
    lowr3 = create_named_configuration("LOWBD2", rmax=3_000.0)

    print("Allocating visibilities")
    vis = create_visibility(
        config=lowr3,
        times=np.linspace(-4, 4, 540) * np.pi / 12,
        frequency=np.r_[15e7],
        channel_bandwidth=np.r_[5e4],
        weight=1.0,
        phasecentre=phasecentre.icrs,
        polarisation_frame=PolarisationFrame("stokesI"),
        times_are_ha=False,
    )

    print("Loading sky GT")
    # Load fits file
    hvox_path = os.sep.join(os.path.abspath(hvox.__file__).split(os.sep)[:-3])
    fits_path = os.path.join(
        hvox_path, "data", "models", "haslam408_dsds_Remazeilles2014.fits"
    )
    hdulist = fits.open(fits_path)
    # Get HPIX metadata
    nside = hdulist[1].header["NSIDE"]
    order = hdulist[1].header["ORDERING"]

    # Create healpix object
    haslam_hpix = HEALPix(nside=nside, order=order, frame=Galactic())
    # get cone of interest in HPIX

    haslam_coi = haslam_hpix.cone_search_lonlat(lon=phasecentre.l, lat=phasecentre.b, radius=radius)
    # get center of the cone (for psf)
    haslam_center = hp.pixelfunc.ang2pix(nside, phasecentre.l.value, phasecentre.b.value, nest=False, lonlat=True)

    # get (lon and lat) coordinates of cone of interest
    directions = haslam_hpix.healpix_to_skycoord(haslam_coi)
    directions_center = haslam_hpix.healpix_to_skycoord(haslam_center)

    # convert to lmn
    lmn = np.asarray(skycoord_to_lmn(directions, phasecentre)).T
    lmn_center = np.asarray(skycoord_to_lmn(directions_center, phasecentre)).squeeze()

    # get map values
    haslam_full = hdulist[1].data['temperature'].ravel()

    if sparse_sky:
        # randomly add ones in an empty sky
        haslam_full = hp.read_map(fits_path)
        haslam_full *= 0
        haslam_full[haslam_coi[np.random.permutation(len(haslam_coi))][:1000]] = 1.

    # get map values at cone of interest
    haslam = haslam_full[haslam_coi]

    print("Simulating visibilities")
    vis_estimate = hvox.dirty2vis(
        uvw=vis.visibility_acc.uvw_lambda.reshape(-1, 3),
        xyz=lmn.reshape(-1, 3),
        dirty=haslam.reshape(-1),
        wgt=vis.visibility_acc.flagged_weight.reshape(-1),
        mesh="hpix",
        normalisation="xyz",
        chunked=True,
    )

    print("Estimating dirty image")
    sky_estimate = hvox.vis2dirty(
        uvw=vis.visibility_acc.uvw_lambda.reshape(-1, 3),
        xyz=lmn.reshape(-1, 3),
        vis=vis_estimate.reshape(-1),
        mesh="hpix",
        wgt=vis.visibility_acc.flagged_weight.reshape(-1),
        normalisation="xyz",
        chunked=True,
    )

    print("Estimating PSF")
    psf_estimate = hvox.compute_psf(
        uvw=vis.visibility_acc.uvw_lambda.reshape(-1, 3),
        xyz=lmn.reshape(-1, 3),
        xyz_center=lmn_center,
        mesh="hpix",
        wgt=vis.visibility_acc.flagged_weight.reshape(-1),
        normalisation="both",
        chunked=True,
    )

    # fig, axs = plt.subplots(
    #     1,
    #     2,
    #     figsize=(10, 4),
    #     subplot_kw={"projection": "3d"},
    # )
    # ids = np.random.permutation(len(lmn))[:1000]
    # axs[0].scatter(*lmn[ids].T, alpha=0.1)
    # axs[0].set_xlabel("x")
    # axs[0].set_ylabel("y")
    # axs[0].set_zlabel("z")
    # uvw = vis.visibility_acc.uvw_lambda.reshape(-1, 3)
    # ids = np.random.permutation(len(uvw))[:1000]
    # axs[1].scatter(*uvw[ids].T, alpha=0.1)
    # axs[1].set_xlabel("u")
    # axs[1].set_ylabel("v")
    # axs[1].set_zlabel("w")
    # plt.show()

    haslam_plot = hp.ma(haslam_full)
    haslam_plot.mask = np.ones_like(haslam_full, dtype=bool)
    haslam_plot.mask[haslam_coi] = False

    dirty_plot = haslam_plot.copy()
    dirty_plot[haslam_coi] = sky_estimate

    psf_plot = haslam_plot.copy()
    psf_plot[haslam_coi] = psf_estimate

    hp.mollview(
        abs(haslam_plot),
        title="Ground Truth",
    )
    hp.graticule()
    plt.show()

    hp.mollview(
        abs(dirty_plot),
        title="Dirty",
    )
    hp.graticule()
    plt.show()

    hp.mollview(
        abs(psf_plot),
        title="PSF",
    )
    hp.graticule()
    plt.show()

    print("Done!")
