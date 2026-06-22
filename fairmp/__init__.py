"""fairmp: fair multimodal meeting-point experiments.

Reference implementation of the algorithm in the paper: a variance-minimising,
H3-indexed coarse-to-fine search for a fair meeting point. The travel-time backend
is abstract: EuclideanBackend runs with no data (for testing), R5Backend uses real
OSM + GTFS via r5py (for results).
"""
__all__ = ["geo", "metrics", "travel_time", "candidates", "algorithm", "baselines", "scenarios", "runner"]
