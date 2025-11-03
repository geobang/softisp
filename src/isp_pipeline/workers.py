"""
Compatibility workers module for convenience importing.

It reuses the per-worker modules and exposes convenience functions:
- fast_algo_worker -> fastalgo.run
- fast_isp_worker  -> fastisp.run
- slow_algo_worker -> slowalgo.run
- slow_isp_worker  -> slowisp.run
- raw_algo_worker  -> rawalgo.run
- raw_isp_worker   -> rawisp.run
"""
from . import fastalgo, fastisp, slowalgo, slowisp, rawalgo, rawisp

def fast_algo_worker(envelope, model_manager=None):
    return fastalgo.run(envelope, model_manager)

def fast_isp_worker(envelope, model_manager=None):
    return fastisp.run(envelope, model_manager)

def slow_algo_worker(envelope, model_manager=None):
    return slowalgo.run(envelope, model_manager)

def slow_isp_worker(envelope, model_manager=None):
    return slowisp.run(envelope, model_manager)

def raw_algo_worker(envelope, model_manager=None):
    return rawalgo.run(envelope, model_manager)

def raw_isp_worker(envelope, model_manager=None):
    return rawisp.run(envelope, model_manager)
