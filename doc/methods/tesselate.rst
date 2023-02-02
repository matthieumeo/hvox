Finding a good partition
========================

HVOX's ability to scale hinges on finding a good partition of :math:`(\Theta,\Omega)` such that sub-problems do useful
computations overall, i.e., that Heisenberg meshes be small and densely-filled.

This in turn implies Heisenberg box dimensions :math:`(\mathcal{B}_{j}, \mathcal{P}_{i})` for each sub-problem
:math:`(i,j)` must be chosen with care:

1) the FFT mesh size within memory constraints, while

2) keeping the number of sub-blocks low to minimize the cost of block assembly, and

3) ensuring the number of visibilities in each sub-block is balanced to have similar execution times, avoiding stalls before block assembly.

Algorithm 3.2 (Find maximum Heisenberg dimensions) in paper [HVOXpaper]_ shows how to cast these objectives as a
6-variable convex optimization problem. The former is easily solved using a linear program (LP), which is cheap and fast.

.. warning::
    The partitioning algorithm is a data-independent method, hence cannot guarantee that condition *(3)* holds.
    Sub-problems :math:`(\Theta_{i},\Omega_{j})` thus obtained may have unbalanced runtimes.
    In practice the [FINUFFT]_ library used in `hvox` solves this issue automatically.

