"""
[LEVEL 99] GENETIC EVOLUTION ENGINE
===================================
Institutional Grade Genetic Algorithm for Strategy Rule Discovery.
Focuses on evolving the 'Logic Genes' (Weights of different signals) rather than just hyperparameters.

Chromosomes:
- [w_trend, w_mean_rev, w_p_win, w_risk_reward, w_ofi, w_impact]
"""

import logging
import json
import os
import random
import copy
import numpy as np
from typing import List, Dict

logger = logging.getLogger("FEAT.Evolution")

class EvolutionEngine:
    def __init__(self, population_size=50, mutation_rate=0.15):
        self.pop_size = population_size
        self.mutation_rate = mutation_rate
        self.gene_pool_path = "data/gene_pool.jsonl"
        
        # Gene Definition: Name -> (Min, Max)
        self.genome_map = {
            "weight_trend": (0.0, 2.0),
            "weight_mean_reversion": (0.0, 2.0),
            "weight_volatility": (-1.0, 1.0),
            "weight_ofi_pressure": (0.5, 3.0),   # Microstructure bias
            "weight_impact_vacuum": (1.0, 4.0),  # Prioritize vacuums
            "stop_loss_decay": (0.9, 1.0)        # Trailing aggression
        }
        
    def run_generation(self, fitness_map: Dict[str, float]):
        """
        Evolves the population based on provided fitness scores.
        fitness_map: { 'dna_hash': sharpe_ratio }
        """
        current_pop = self._load_population()
        if not current_pop:
            logger.info("🧬 Genesis: Creating Initial Population.")
            current_pop = [self._random_individual() for _ in range(self.pop_size)]
        
        # Assign fitness
        scored_pop = []
        for ind in current_pop:
            dna_hash = self._hash_dna(ind)
            score = fitness_map.get(dna_hash, -10.0) # Default penalty for untested
            scored_pop.append((ind, score))
            
        # Elitism: Keep Top 2
        scored_pop.sort(key=lambda x: x[1], reverse=True)
        next_gen = [x[0] for x in scored_pop[:2]]
        
        logger.info(f"🧬 Evolution: Best Fitness = {scored_pop[0][1]:.4f}")
        
        # Breeding Loop
        while len(next_gen) < self.pop_size:
            p1 = self._tournament_select(scored_pop)
            p2 = self._tournament_select(scored_pop)
            
            child = self._crossover(p1, p2)
            child = self._mutate(child)
            next_gen.append(child)
            
        self._save_population(next_gen)
        return next_gen[0] # Return Alpha for deployment

    def _random_individual(self) -> Dict:
        return {k: round(random.uniform(v[0], v[1]), 3) for k, v in self.genome_map.items()}

    def _tournament_select(self, scored_pop):
        tournament = random.sample(scored_pop, k=3)
        return max(tournament, key=lambda x: x[1])[0]

    def _crossover(self, p1: Dict, p2: Dict) -> Dict:
        child = {}
        for k in p1:
            child[k] = p1[k] if random.random() > 0.5 else p2[k]
        return child

    def _mutate(self, ind: Dict) -> Dict:
        mutant = copy.deepcopy(ind)
        for k, bounds in self.genome_map.items():
            if random.random() < self.mutation_rate:
                # Gaussian Mutation
                sigma = (bounds[1] - bounds[0]) * 0.1
                delta = random.gauss(0, sigma)
                mutant[k] = max(bounds[0], min(bounds[1], mutant[k] + delta))
                mutant[k] = round(mutant[k], 3)
        return mutant

    def _hash_dna(self, dna: Dict) -> str:
        s = json.dumps(dna, sort_keys=True)
        return str(hash(s))

    def _load_population(self) -> List[Dict]:
        if not os.path.exists(self.gene_pool_path): return []
        try:
            with open(self.gene_pool_path, "r") as f:
                return [json.loads(line) for line in f]
        except: return []

    def _save_population(self, pop: List[Dict]):
        os.makedirs("data", exist_ok=True)
        with open(self.gene_pool_path, "w") as f:
            for ind in pop:
                f.write(json.dumps(ind) + "\n")
                
# Singleton
evolution_engine = EvolutionEngine()
