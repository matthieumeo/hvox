HVOX
====

In radio astronomy, an interferometer samples the complex visibily function:

.. math::
    V_{i} = \int_{\mathbb{S}^{2}} I(\mathbf{r}) \exp\left(-j \left<\mathbf{r}, \, \mathbf{p}_{i}\right>\right)\text{d}\mathbf{r},

also known as the **measurement equation**, where :math:`I(\mathbf{r}) \in \mathbb{S}_{+}` denotes the spherical sky function, 
:math:`\mathbf{p}_{i} \in \mathbb{S}^{3}` the :math:`i`-th normalized baseline, and :math:`V_{i} \in \mathbb{C}` the measured visibility.

Once discretized, the measurement equation gives the **analysis** equation and its adjoint **synthesis** equations:

- Synthesis:

.. math::
    V_{i} = \sum_{\mathbf{r} \in \Theta_{\text{pix}}} I(\mathbf{r}) \, \alpha_{\Theta}(\mathbf{r}) \, \exp\left(-j \left<\mathbf{r},\, \mathbf{p}_{i}\right>\right) \\

- Analysis:

.. math::
    I(\mathbf{r}) = \sum_{i = 1}^{N_{\text{vis}}} V_{i} \, \alpha_{\Theta}(\mathbf{r}) \, \exp\left(j \left<\mathbf{r},\,\mathbf{p}_{i}\right>\right)

Where :math:`\Theta_{\text{pix}}` denotes some discrete tesselation of the sphere :math:`\mathbb{S}^{2}`,
:math:`\alpha_{\Theta}(\mathbf{r}) \in \mathbb{R}` a weighting function due to discretizing the measurement equation,
and :math:`N_{\text{vis}}` the number of measured visibilities.


The HVOX method performs the **analysis** and **synthesis** via the 3D non-uniform Fast Fourier Transform of type 3
(3D-NUFFT3), leveraging the [FINUFFT]_ Python 3 package.
