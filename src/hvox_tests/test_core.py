import numpy as np

try:
    import pytest
    import ducc0.wgridder.experimental as wgridder
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
    raise ValueError("Missing packages, install `test` requirements.")

import hvox


class TestHVOX_dcos:
    @pytest.fixture
    def gt(self):
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
        model = create_test_image(
            phasecentre=vis.phasecentre, frequency=np.r_[15e7], cellsize=5e-4
        )

        direction_cosines, jacobian = get_direction_cosines(model)
        wgt_dirty = 1 / jacobian.reshape(-1)

        return vis, model, direction_cosines, wgt_dirty

    @pytest.fixture(params=["single", "double"])
    def dtype(self, request):
        dtype_r = np.single if request.param == "single" else np.double
        dtype_c = np.csingle if request.param == "single" else np.cdouble
        return [dtype_r, dtype_c]

    @pytest.fixture
    def vis(self, gt, dtype):
        visibilities = gt[0].copy()
        visibilities.vis.data = visibilities.vis.data.astype(dtype[1])
        return visibilities

    @pytest.fixture
    def model(self, gt, dtype):
        sky_model = gt[1]
        sky_model.pixels.data = sky_model.pixels.data.astype(dtype[0])
        return sky_model

    @pytest.fixture
    def direction_cosines(self, gt):
        return gt[2]

    @pytest.fixture
    def wgt_dirty(self, gt, dtype):
        return gt[3].astype(dtype[0])

    @pytest.fixture(params=["w-True", "w-False"])
    def w_term(self, request):
        if request.param == "w-True":
            return True
        else:
            return False

    @pytest.fixture
    def epsilon(self, dtype, w_term):
        if w_term:
            if np.issubdtype(dtype[0], np.single):
                return 1e-5
            else:
                return 1e-9
        else:
            return 1e-3
    @pytest.fixture(params=["ch-True", "ch-False"])
    def chunked(self, request):
        if request.param == "ch-True":
            return True
        else:
            return False

    @pytest.fixture
    def hvox_results(
        self, vis, direction_cosines, model, wgt_dirty, w_term, epsilon, chunked,
    ):
        vis_hvox = hvox.dirty2vis(
            uvw_lambda=vis.visibility_acc.uvw_lambda.reshape(-1, 3),
            xyz=direction_cosines.reshape(-1, 3),
            dirty=model.pixels.data.reshape(-1),
            wgt_vis=vis.visibility_acc.flagged_weight.reshape(-1),
            wgt_dirty=wgt_dirty,
            w_term=w_term,
            epsilon=epsilon,
            chunked=chunked,
        )

        dirty_hvox = hvox.vis2dirty(
            uvw_lambda=vis.visibility_acc.uvw_lambda.reshape(-1, 3),
            xyz=direction_cosines.reshape(-1, 3),
            vis=vis_hvox.reshape(-1),
            wgt_vis=None,
            wgt_dirty=wgt_dirty.astype(model.pixels.data.dtype),
            w_term=w_term,
            epsilon=epsilon,
            chunked=chunked,
        )
        return vis_hvox, dirty_hvox

    @pytest.fixture
    def ducc0_results(
            self,
            vis, model,
            epsilon, w_term,
    ):
        npixels = model.pixels.shape[-1]
        pixsize = np.abs(np.radians(model.image_acc.wcs.wcs.cdelt[0]))
        fuvw = vis.uvw.data.copy().reshape(-1, 3)
        fuvw[:, 0] *= -1.0
        fuvw[:, 2] *= -1.0
        vis_ducc0 = wgridder.dirty2vis(
            uvw=fuvw,
            freq=vis.frequency.data,
            dirty=model.pixels.data.squeeze().T,
            wgt=vis.visibility_acc.flagged_weight.reshape(-1, 1).astype(model.pixels.data.dtype),
            pixsize_x=pixsize,
            pixsize_y=pixsize,
            epsilon=epsilon,
            do_wgridding=w_term,
            nthreads=4,
            verbosity=0,
        )

        dirty_ducc0 = wgridder.vis2dirty(
            uvw=fuvw,
            freq=vis.frequency.data,
            vis=vis_ducc0.reshape(-1, 1),
            npix_x=npixels,
            npix_y=npixels,
            pixsize_x=pixsize,
            pixsize_y=pixsize,
            epsilon=epsilon,
            do_wgridding=w_term,
            nthreads=4,
            verbosity=0,
        )
        return vis_ducc0, dirty_ducc0

    def test_dirty2vis(self, ducc0_results, hvox_results, epsilon):
        assert np.linalg.norm(ducc0_results[0].ravel() - hvox_results[0].ravel()) / np.linalg.norm(ducc0_results[0]) < epsilon


    def test_vis2dirty(self, ducc0_results, hvox_results, epsilon):
        assert np.linalg.norm(ducc0_results[1].ravel() - hvox_results[1].ravel()) / np.linalg.norm(ducc0_results[1]) < epsilon

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