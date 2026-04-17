API Reference
=============

Optimizers
----------

.. autoclass:: torch_dfo.CMAES
   :members: ask, tell, best, state_dict, load_state_dict, restart

.. autoclass:: torch_dfo.SHADE
   :members: ask, tell, best, state_dict, load_state_dict

.. autoclass:: torch_dfo.NelderMead
   :members: ask, tell, best, state_dict, load_state_dict

.. autoclass:: torch_dfo.PhasedDFO
   :members: ask, tell, best, state_dict, load_state_dict, phase, fe_count, done, budget

.. autoclass:: torch_dfo.dlr_cma.DLRPortfolio
   :members: ask, tell, state_dict, load_state_dict

Wrappers
--------

.. autoclass:: torch_dfo.DFOOptimizer
   :members:

Base
----

.. autoclass:: torch_dfo.BaseOptimizer
   :members: ask, tell, best, state_dict, load_state_dict

Search Space
------------

.. autoclass:: torch_dfo.SearchSpace
   :members:

.. autoclass:: torch_dfo.Float
   :members:

.. autoclass:: torch_dfo.Int
   :members:

.. autoclass:: torch_dfo.Categorical
   :members:

Benchmarks
----------

.. automodule:: torch_dfo.benchmarks
   :members:
