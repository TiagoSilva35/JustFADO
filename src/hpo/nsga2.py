"""Lightweight NSGA-II utilities for multi-objective parameter tuning."""

from __future__ import annotations

import random

from typing import Callable, Dict, List, Tuple
import copy
from src.helpers import utils
import numpy as np


Candidate = Dict[str, float]
Objectives = Tuple[float, ...]


def _dominates(a: Objectives, b: Objectives) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def _fast_non_dominated_sort(objectives: List[Objectives]) -> List[List[int]]:
    domination_counts = [0] * len(objectives)
    dominates = [[] for _ in objectives]
    fronts: List[List[int]] = [[]]

    for i in range(len(objectives)):
        for j in range(len(objectives)):
            if i == j:
                continue
            if _dominates(objectives[i], objectives[j]):
                dominates[i].append(j)
            elif _dominates(objectives[j], objectives[i]):
                domination_counts[i] += 1
        if domination_counts[i] == 0:
            fronts[0].append(i)

    k = 0
    while k < len(fronts) and fronts[k]:
        next_front = []
        for i in fronts[k]:
            for j in dominates[i]:
                domination_counts[j] -= 1
                if domination_counts[j] == 0:
                    next_front.append(j)
        k += 1
        fronts.append(next_front)

    return fronts[:-1]


def _crowding_distance(front: List[int], objectives: List[Objectives]) -> Dict[int, float]:
    if not front:
        return {}
    distances = {idx: 0.0 for idx in front}
    n_obj = len(objectives[0])

    for m in range(n_obj):
        ordered = sorted(front, key=lambda idx: objectives[idx][m])
        distances[ordered[0]] = float("inf")
        distances[ordered[-1]] = float("inf")
        min_v = objectives[ordered[0]][m]
        max_v = objectives[ordered[-1]][m]
        denom = max(max_v - min_v, 1e-12)
        for i in range(1, len(ordered) - 1):
            left_v = objectives[ordered[i - 1]][m]
            right_v = objectives[ordered[i + 1]][m]
            distances[ordered[i]] += (right_v - left_v) / denom
    return distances


def _sample(bounds: Dict[str, Tuple[float, float]], rng: random.Random) -> Candidate:
    return {k: float(rng.uniform(low, high)) for k, (low, high) in bounds.items()}


def _crossover(a: Candidate, b: Candidate, rng: random.Random) -> Candidate:
    child: Candidate = {}
    for k in a:
        alpha = rng.random()
        child[k] = alpha * a[k] + (1.0 - alpha) * b[k]
    return child


def _mutate(child: Candidate, bounds: Dict[str, Tuple[float, float]], rng: random.Random,
            mutation_rate: float = 0.2) -> Candidate:
    out = dict(child)
    for k, (low, high) in bounds.items():
        if rng.random() < mutation_rate:
            span = high - low
            out[k] = min(high, max(low, out[k] + rng.uniform(-0.15 * span, 0.15 * span)))
    return out


def _select_parent(pop: List[Candidate], objectives: List[Objectives], rng: random.Random) -> Candidate:
    i, j = rng.randrange(len(pop)), rng.randrange(len(pop))
    oi, oj = objectives[i], objectives[j]
    if _dominates(oi, oj):
        return pop[i]
    if _dominates(oj, oi):
        return pop[j]
    return pop[i] if rng.random() < 0.5 else pop[j]


def _choose_compromise(front: List[int], objectives: List[Objectives]) -> int:
    front_objs = np.array([objectives[i] for i in front], dtype=float)
    mins = np.min(front_objs, axis=0)
    maxs = np.max(front_objs, axis=0)
    denom = np.maximum(maxs - mins, 1e-12)
    scores = np.sum((front_objs - mins) / denom, axis=1)
    return front[int(np.argmin(scores))]


def run_nsga2(
    objective_fn: Callable[[Candidate], Objectives],
    bounds: Dict[str, Tuple[float, float]],
    population_size: int = 8,
    generations: int = 4,
    seed: int = 42,
) -> Tuple[Candidate, Objectives, List[Candidate], List[Objectives]]:
    """Run compact NSGA-II and return a compromise solution from the first Pareto front."""
    rng = random.Random(seed)
    population = [_sample(bounds, rng) for _ in range(population_size)]
    objectives = [objective_fn(c) for c in population]

    for _ in range(generations):
        offspring: List[Candidate] = []
        while len(offspring) < population_size:
            p1 = _select_parent(population, objectives, rng)
            p2 = _select_parent(population, objectives, rng)
            child = _mutate(_crossover(p1, p2, rng), bounds, rng)
            offspring.append(child)

        off_objectives = [objective_fn(c) for c in offspring]
        combined = population + offspring
        combined_obj = objectives + off_objectives

        fronts = _fast_non_dominated_sort(combined_obj)
        next_population: List[Candidate] = []
        next_objectives: List[Objectives] = []

        for front in fronts:
            if len(next_population) + len(front) <= population_size:
                for idx in front:
                    next_population.append(combined[idx])
                    next_objectives.append(combined_obj[idx])
            else:
                crowd = _crowding_distance(front, combined_obj)
                ordered = sorted(front, key=lambda idx: crowd[idx], reverse=True)
                remaining = population_size - len(next_population)
                for idx in ordered[:remaining]:
                    next_population.append(combined[idx])
                    next_objectives.append(combined_obj[idx])
                break

        population, objectives = next_population, next_objectives

    final_fronts = _fast_non_dominated_sort(objectives)
    first_front = final_fronts[0] if final_fronts else list(range(len(population)))
    best_idx = _choose_compromise(first_front, objectives)
    return population[best_idx], objectives[best_idx], population, objectives

