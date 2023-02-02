Chunking
========

Assuming the 3D Heisenberg mesh fits in memory, the amount of useful work performed by the 3D FFT is low due to the
geometrical structure of baselines and sky tesselations.

Baselines are distributed unevenly in the UVW frame: there is a high density of short baselines near the origin,
and sparse coverage in peripheral regions. The sky coordinates are also distributed unevenly in the XYZ frame: they lie
on a 2D manifold occupying a tiny fraction of the 3D volume. For this reason, most of the FFT effort is thus wasted
given low fill-in of the mesh.

In summary, HVOX produces images directly in any tesselation suitable for image-processing production pipelines,
and has the ability to auto-choose a suitable kernel support for desired accuracy. The 3D-NUFFT3 is performed at a mesh size that is independent of the number of pixels, based on the Heisenberg uncertainty principle, with box dimensions kept as small as possible.


Let :math:`\left\{\Theta_{i}\right\}_{i=1}^{N_{\text{pix}}^{\text{blk}}}` and 
:math:`\left\{\Omega_{j}\right\}_{j=1}^{N_{\text{vis}}^{\text{blk}}}` denote some disjoint partitioning of 
:math:`(\Theta, \Omega)`, and :math:`\mathbf{V}_{\Omega_{j}}` denote the subset of visibilities associated to baselines 
:math:`\Omega_{j}`.

Then the **synthesis** equation can be rewritten as:

.. math::
    \mathbf{I}
    = \sum_{i=1}^{N_{\text{pix}}^{\text{blk}}} \sum_{j=1}^{N_{\text{vis}}^{\text{blk}}} \mathbf{A}_{\Theta_{i},\Omega_{j}} \mathbf{V}_{\Omega_{j}}.


Which shows that a monolithic type-3 transform mapping :math:`\Omega` to :math:`\Theta` can be computed by linear
combination of :math:`N_{\text{vis}}^{\text{blk}} N_{\text{pix}}^{\text{blk}}` type-3 sub-transforms mapping
:math:`\Omega_{j}` to :math:`\Theta_{i}`.

Evaluating **synthesis** via the splitting or chunking method provides the following benefits:

- Each sub-transform operates on a subset of visibilities and pixels. If :math:`(\Theta, \Omega)` is well-partitioned then the Heisenberg boxes associated to each sub-transform shrink drastically, and so do their respective FFT meshes. This leads to drastic reduction of FFT memory/compute requirements by not wasting arithmetic on empty baseline/sky regions.

- Since sub-transforms are small and independent, they can be computed in parallel to reduce synthesis time.

- If :math:`\vert\Theta_{i}\vert \times \vert\Omega_{j}\vert` is small, i.e. the sub-transform :math:`(i,j)` is transforming a region in baseline-space and/or sky-space with low density, then it is faster to evaluate :math:`\mathbf{A}_{\Theta_{i},\Omega_{j}} V_{\Omega_{j}}` via direct summation rather than a type-3 NUFFT. Splitting allows one to tailor the synthesis process at the block level based on its characteristics.