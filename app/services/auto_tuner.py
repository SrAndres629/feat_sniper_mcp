
import logging
import json
import os
import random
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger("FEAT.AutoTuner")

class AutoTuner:
    """
    [LEVEL 31] GENETIC POPULATION OPTIMIZER
    Implements a full Genetic Algorithm (Population, Crossover, Selecting)
    using 'Virtual Replay' on Experience Memory to evaluate fitness.
    
    Replaces legacy Bayesian Optimizer.
    """
    
    def __init__(self, memory_path: str = "data/experience_memory.jsonl"):
        self.memory_path = memory_path
        # [USER REQUEST] Ensure write permission to config used by system
        self.output_path = "config/dynamic_params.json"
        
        self.population_size = 20
        self.generations = 5
        self.mutation_rate = 0.2
        
        # Base Gene Pool (Ranges)
        self.gene_bounds = {
            "ALPHA_CONFIDENCE_THRESHOLD": (0.55, 0.85),
            "ATR_TRAILING_MULTIPLIER": (1.0, 3.5),
            "RISK_PER_TRADE_PERCENT": (0.5, 2.0)
        }
        
    def evolve_population(self):
        """
        Runs the Genetic Algorithm Cycle.
        1. Load History (Replay Buffer)
        2. Initialize Population
        3. Iterate Generations (Select -> Crossover -> Mutate)
        4. Save Alpha (Best Individual)
        """
        history = self._load_experience()
        if len(history) < 20: # Need decent sample size
            # Fallback to heuristic or do nothing
            logger.info("🧬 GA: Not enough data for population sim (Need >20).")
            return

        logger.info(f"🧬 GA: Starting Evolution on {len(history)} experiences...")
        
        # 1. Initialize Population
        population = [self._random_individual() for _ in range(self.population_size)]
        
        best_overall = None
        best_fitness = -float('inf')
        
        # 2. Generations
        for gen in range(self.generations):
            # Evaluate Fitness
            fitness_scores = [self._evaluate_fitness(ind, history) for ind in population]
            
            # Track Best
            gen_best_idx = np.argmax(fitness_scores)
            gen_best = population[gen_best_idx]
            gen_fitness = fitness_scores[gen_best_idx]
            
            if gen_fitness > best_fitness:
                best_fitness = gen_fitness
                best_overall = gen_best
                
            logger.info(f"🧬 Gen {gen+1}: Best Fitness = {gen_fitness:.4f}")
            
            # Select Parents (Tournament)
            parents = self._selection(population, fitness_scores)
            
            # Crossover & Mutation
            next_generation = []
            while len(next_generation) < self.population_size:
                p1, p2 = random.sample(parents, 2)
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                next_generation.append(child)
                
            population = next_generation
            
        # 3. Save Winner
        if best_overall:
            logger.info(f"🏆 Evolution Complete. Winner: {best_overall}")
            self._save_genes(best_overall)

    def _evaluate_fitness(self, genes: Dict, history: List[Dict]) -> float:
        """
        Virtual Replay: Simulates PnL if these genes were active.
        """
        simulated_pnl = 0.0
        trades_taken = 0
        
        threshold = genes["ALPHA_CONFIDENCE_THRESHOLD"]
        risk = genes["RISK_PER_TRADE_PERCENT"]
        stop_mult = genes["ATR_TRAILING_MULTIPLIER"]
        
        for record in history:
            # Reconstruct State (Idealized)
            snapshot = record.get("state_snapshot", {})
            
            # We use recorded 'confidence' or 'lstm_prob'
            original_confidence = snapshot.get("confidence", 0.6)
            
            # Would we trade?
            if original_confidence >= threshold:
                # Yes, we trade.
                # Outcome logic:
                # If we have 'pnl' in record, that's the result of the ORIGINAL strategy.
                # To simulate NEW strategy result, we need candle data (hard).
                # SImplification: We assume the 'direction' is correct if PnL > 0.
                
                # If original PnL > 0 -> We win. Scale by risk.
                # If original PnL < 0 -> We lose. Scale by risk.
                
                # BUT, tighter stops (ATR Mult) might mitigate loss or cut win short.
                # This is a heuristic simulation.
                
                pnl = record.get("pnl", 0.0)
                
                # Apply Risk Scaling
                scaled_pnl = pnl * risk 
                
                simulated_pnl += scaled_pnl
                trades_taken += 1
            else:
                # We skipped this trade. PnL 0.
                pass
                
        # Fitness Function
        if trades_taken == 0: return -1.0
        return simulated_pnl

    def _random_individual(self) -> Dict:
        return {
            k: round(random.uniform(v[0], v[1]), 2)
            for k, v in self.gene_bounds.items()
        }
        
    def _selection(self, population, fitness_scores):
        # Tournament Selection
        selected = []
        k = 3 # Tournament size
        for _ in range(len(population) // 2): # Select half as parents
            candidates_idx = random.sample(range(len(population)), k)
            best_cand_idx = max(candidates_idx, key=lambda i: fitness_scores[i])
            selected.append(population[best_cand_idx])
        return selected

    def _crossover(self, p1: Dict, p2: Dict) -> Dict:
        # Uniform Crossover
        child = {}
        for k in p1.keys():
            child[k] = p1[k] if random.random() > 0.5 else p2[k]
        return child

    def _mutate(self, ind: Dict) -> Dict:
        child = ind.copy()
        for k, v in self.gene_bounds.items():
            if random.random() < self.mutation_rate:
                # Mutate by small delta (+- 10% of range)
                delta = (v[1] - v[0]) * 0.1 * random.uniform(-1, 1)
                new_val = child[k] + delta
                # Clamp
                child[k] = max(v[0], min(v[1], new_val))
                child[k] = round(child[k], 2)
        return child
        
    def _load_experience(self) -> List[Dict]:
        if not os.path.exists(self.memory_path): return []
        data = []
        try:
            with open(self.memory_path, "r") as f:
                for line in f:
                    try:
                        data.append(json.loads(line))
                    except: pass
        except: pass
        return data[-200:] # Use last 200

    def _save_genes(self, genes: Dict):
        # Writes strictly to config/dynamic_params.json for System-wide Adoption
        os.makedirs("config", exist_ok=True)
        try:
            with open(self.output_path, "w") as f:
                json.dump(genes, f, indent=4)
            logger.info(f"[AUTO-TUNER] Successfully updated {self.output_path}")
        except Exception as e:
            logger.error(f"[AUTO-TUNER] Failed to write config: {e}")

# Singleton Instance
auto_tuner = AutoTuner()

if __name__ == "__main__":
    # Test Run
    auto_tuner.evolve_population()
