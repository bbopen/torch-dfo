Benchmarks and scaling guidance
================================

This page summarises the v0.9 multi-function scaling sweep and the
practical guidance it implies for choosing an optimizer.

All numbers below come from a single NVIDIA RTX A4500 (19.6 GB usable
VRAM) running double-precision (``float64``) workloads over six
classical test functions (sphere, rosenbrock, rastrigin, ackley,
griewank, levy) at ten dimensions between ``d=20`` and ``d=20 480``.

Peak-VRAM ceiling per optimizer
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Optimizer
     - Max dim (no OOM)
     - Notes
   * - ``DLRPortfolio``
     - ≥ 20 480 (ceiling not hit)
     - Constant pop=60. Peak ≤ 123 MB across the sweep.
   * - ``SHADE``
     - ≥ 20 480 (ceiling not hit)
     - Constant pop=80. Peak ≤ 151 MB across the sweep.
   * - ``NelderMead``
     - 20 480
     - Peak 9.6–19.2 GB depending on the objective
       (see `issue #9 <https://github.com/bbopen/torch-dfo/issues/9>`_
       for the high-dim VRAM story).
   * - ``CMAES``
     - 10 240
     - ``O(d²)`` covariance; at ``d=20 480`` the matrix alone is
       ~3.4 GB in ``float64``.
   * - ``PhasedDFO``
     - 5 120
     - Auto-schedule grows pop as ~4·d; see
       `issue #8 <https://github.com/bbopen/torch-dfo/issues/8>`_
       for the cap proposal.

Choosing an optimizer by dimension
----------------------------------

For the 20 GB, double-precision regime the sweep was run in, a
reasonable default policy is:

* ``d ≤ 5 000`` — any optimizer in the library is viable.
* ``5 000 < d ≤ 10 000`` — ``DLRPortfolio``, ``SHADE``, or
  ``NelderMead``.
* ``d > 10 000`` — ``DLRPortfolio`` or ``SHADE``.

The ceiling scales roughly with VRAM, so a 40 GB card pushes each
range up by roughly one doubling of ``d``.

Solution quality at low dim
---------------------------

.. warning::

   This is not a fair comparison. The ``d=40`` slice below fixes
   *wall-clock* (7 generations), not evaluation count. Within that
   budget, ``CMAES`` ran ~105 fevals, ``DLRPortfolio`` ~420, ``SHADE``
   ~560, and ``PhasedDFO`` up to ~137 000. A fevals-equalized race
   would change the picture; read the numbers below as "what a user
   gets if they cap wall-clock", not as an algorithm ranking.

Separately from ceiling, the ``d=40`` slice of the sweep — a 7-generation
wall-clock budget — shows ``DLRPortfolio`` reaching one to three orders
of magnitude lower ``final_loss`` than the other four optimizers on
every multimodal function (ackley, griewank, rastrigin, levy).

Reproducing the sweep
---------------------

The sweep lives in ``benchmarks/run_scaling_sweep.py`` and its
underlying probe in ``benchmarks/scaling_probe.py``. A single run
covering six functions, five optimizers, and the full dim ladder takes
roughly 45 minutes on an RTX A4500::

    python benchmarks/run_scaling_sweep.py \
        --functions sphere rosenbrock rastrigin ackley griewank levy \
        --optimizers DLRPortfolio SHADE CMAES NelderMead PhasedDFO \
        --warmup 2 --timed 5

See `issue #10 <https://github.com/bbopen/torch-dfo/issues/10>`_ for the
proposed default ``--warmup`` change.
