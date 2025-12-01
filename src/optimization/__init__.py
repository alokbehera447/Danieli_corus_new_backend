"""Optimization module for genetic algorithm packing optimization."""

from .fitness_function import (
    PenaltyWeights,
    DEFAULT_WEIGHTS,
    Gene,
    Individual,
    FitnessResult,
    FitnessEvaluator,
    evaluate_packing_fitness,
    count_overlapping_pairs,
    count_out_of_bounds,
    estimate_num_cuts
)
