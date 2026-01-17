
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
        
        # Base Gene Pool (Ranges) - [LEVEL 34] EXPANDED ANALYSIS & MANAGEMENT
        self.gene_bounds = {
            "ALPHA_CONFIDENCE_THRESHOLD": (0.60, 0.90),
            "ATR_TRAILING_MULTIPLIER": (1.5, 4.0),
            "ATR_SL_MULTIPLIER": (1.0, 3.0),
            "ATR_TP_MULTIPLIER": (2.0, 6.0),
            "RISK_PER_TRADE_PERCENT": (0.5, 2.0),
            "COMPOUND_SHARE": (0.3, 0.7) # 30% to 70% share for reinvestment
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
            
            # Broadcast Generation Progress to Dashboard
            self._broadcast_progress(
                gen=gen+1,
                best_fitness=gen_fitness,
                population_size=self.population_size,
                status="TRAINING",
                message=f"Generation {gen+1} complete. Best Fitness: {gen_fitness:.4f}"
            )
            
        # 3. Save Winner
        if best_overall:
            logger.info(f"🏆 Evolution Complete. Winner: {best_overall}")
            self._broadcast_progress(
                gen=self.generations,
                best_fitness=best_fitness,
                best_dna=best_overall,
                status="COMPLETED",
                message=f"🏆 Evolution Successful! DNA Optimized: {list(best_overall.keys())}"
            )
            self._save_genes(best_overall)

    def _evaluate_fitness(self, genes: Dict, history: List[Dict]) -> float:
        """
        [SENIOR RLAIF] Virtual Replay with Constitutional Penalties.
        Evaluates not just PnL, but quality of execution vs Physics.
        """
        total_score = 0.0
        trades_taken = 0
        winning_trades = 0
        
        threshold = genes["ALPHA_CONFIDENCE_THRESHOLD"]
        risk = genes["RISK_PER_TRADE_PERCENT"]
        
        # Ranges from genes (if optimized)
        # Note: In a real simulation, these would change the outcome of entries/exits.
        # Here we use them as heuristics on existing history.
        
        for record in history:
            # We look for RLAIF JUDGMENT or raw experience
            snapshot = record.get("state_snapshot", {})
            pnl = record.get("pnl", 0.0)
            
            # 1. Extraction of Constitutional Metrics
            # If the record is a result of a previous Critic run, it has these.
            # Otherwise we look at the snapshot.
            original_confidence = snapshot.get("confidence", record.get("lstm_conf", 0.6))
            feat_score = snapshot.get("feat_score", record.get("feat_score", 50.0))
            acceleration = snapshot.get("acceleration", record.get("acceleration", 0.0))
            
            # 2. Entry Decision (Neural Gating)
            if original_confidence >= threshold:
                trades_taken += 1
                
                # 3. RLAIF FITNESS LOGIC
                # Basic PnL scaled by risk
                trade_reward = pnl * risk
                
                # PENALTY: Luck-based gain (Bad Process + Good Outcome)
                if pnl > 0 and feat_score < 60:
                    # High penalty for winning by chance/noise
                    trade_reward *= 0.2
                    logger.debug(f"⚠️ GA: Luck-based trade penalized (Score: {feat_score})")
                
                # PENALTY: Physics Violation (Low acceleration)
                if abs(acceleration) < 0.1:
                    trade_reward -= 5.0 # Flat penalty for trading dead markets
                
                # BONUS: Structural Integrity (Good Process + Good Outcome)
                if pnl > 0 and feat_score >= 75:
                    trade_reward *= 1.5 
                
                # FATAL: Trading against Macro Gravity (Simplified check if available)
                # if record.get("trend_aligned") == False: trade_reward -= 10.0

                total_score += trade_reward
                if pnl > 0: winning_trades += 1
        
        # 4. Global Performance Metrics (Profit / Hour Ratio equivalent)
        if trades_taken == 0:
            return -100.0 # Heavy penalty for too defensive genes
            
        win_rate = winning_trades / trades_taken
        
        # Fitness combines PnL, WinRate and Stability
        # We want a positive gain per hour ratio
        fitness = total_score / (len(history) / 100.0) # Normalized
        
        # Penalize low win-rate if below survival threshold
        if win_rate < 0.35:
            fitness *= 0.5
            
        return round(fitness, 4)

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
                    except Exception: 
                        _corrupt_line = True
        except: pass
        return data[-10000:] # Increased for Mass Training (180 days)

    def _save_genes(self, genes: Dict):
        # Writes strictly to config/dynamic_params.json for System-wide Adoption
        os.makedirs("config", exist_ok=True)
        try:
            with open(self.output_path, "w") as f:
                json.dump(genes, f, indent=4)
            logger.info(f"[AUTO-TUNER] Successfully updated {self.output_path}")
        except Exception as e:
            logger.error(f"[AUTO-TUNER] Failed to write config: {e}")

    def _broadcast_progress(self, gen: int, best_fitness: float, population_size: int = 20, 
                           best_dna: Dict = None, status: str = "IDLE", message: str = ""):
        """Sends GA metrics to Supabase for the Neural Evolution Dashboard."""
        try:
            from app.services.supabase_sync import supabase_sync
            data = {
                "generation": gen,
                "best_fitness": best_fitness,
                "population_size": population_size,
                "best_dna": best_dna,
                "status": status,
                "message": message
            }
            # We use the neural_evolution table
            # Assuming supabase_sync has an insert method for generic tables or we use client directly
            if hasattr(supabase_sync, 'client') and supabase_sync.client:
                supabase_sync.client.table("neural_evolution").insert(data).execute()
                logger.debug(f"[GA_BROADCAST] Status: {status} | Gen: {gen}")
            else:
                logger.warning("[GA_BROADCAST] Supabase Client unavailable.")
        except Exception as e:
            logger.error(f"[GA_BROADCAST] Error: {e}")

# Singleton Instance
auto_tuner = AutoTuner()

if __name__ == "__main__":
    # Test Run
    auto_tuner.evolve_population()
