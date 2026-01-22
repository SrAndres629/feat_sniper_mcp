import argparse
import sys
import os
import subprocess
import logging
from datetime import datetime

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | [COMMANDER] | %(message)s")
logger = logging.getLogger("NexusCommander")

def run_training(epochs=50, gpu=True, batch_size=32, limit=0):
    """
    PHASE 1: THE FORGE
    Launches train_hybrid.py with Sovereign Logic.
    """
    logger.info(f"⚔️ IGNITING THE FORGE: {epochs} Epochs | GPU={gpu}")
    
    cmd = [
        sys.executable, "nexus_training/train_hybrid.py",
        "--real",
        "--symbol", "XAUUSD",
        "--epochs", str(epochs),
        "--batch_size", str(batch_size)
    ]
    
    if limit > 0:
        cmd.extend(["--limit", str(limit)])
        
    try:
        subprocess.run(cmd, check=True)
        logger.info("✅ FORGE COMPLETE. Model Weights Optimized.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ FORGE FAILED. Exit Code: {e.returncode}")
        return False

def run_war_games(episodes=10):
    """
    PHASE 2: WAR GAMES
    Launches simulate_warfare.py for RL Punishment.
    """
    logger.info(f"🛡️ LAUNCHING WAR GAMES: {episodes} Episodes")
    cmd = [sys.executable, "nexus_training/simulate_warfare.py", "--episodes", str(episodes)]
    subprocess.run(cmd)

def run_daemon():
    """
    PHASE 3: DAEMON
    Launches the Live Operational Sentinel.
    """
    logger.info("📡 STARTING NEXUS DAEMON (LIVE OPS)...")
    cmd = [sys.executable, "nexus_daemon.py"]
    subprocess.run(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FEAT Sniper Nexus Commander - Sovereign Orchestrator V6",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["train", "war", "daemon", "full_auto"], 
        default="full_auto",
        help="System operation mode: 'train' (Neural Forge), 'war' (RL Refinement), 'daemon' (Live Ops), or 'full_auto' (Pipeline)."
    )
    
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs for the Forge.")
    parser.add_argument("--batch_size", type=int, default=32, help="Internal batch size for gradient updates.")
    parser.add_argument("--limit", type=int, default=0, help="Limit dataset size (0 for full dataset).")
    parser.add_argument("--symbol", type=str, default="XAUUSD", help="Target trading symbol.")
    parser.add_argument("--gpu", action="store_true", help="Force CUDA acceleration if available.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes for War Games (RL).")
    parser.add_argument("--dry-run", action="store_true", help="Audit configuration without engaging engine.")
    
    args = parser.parse_args()
    
    logger.info("🛠️ SOVEREIGN CLI ACTIVATED | Mode: %s", args.mode.upper())
    
    try:
        if args.mode == "train":
            run_training(
                epochs=args.epochs, 
                gpu=args.gpu, 
                batch_size=args.batch_size, 
                limit=args.limit
            )
            
        elif args.mode == "war":
            run_war_games(episodes=args.episodes)
            
        elif args.mode == "daemon":
            run_daemon()
            
        elif args.mode == "full_auto":
            logger.info("🚀 FULL AUTO SEQUENCE INITIATED: [Forge -> War Games -> Daemon]")
            if run_training(epochs=args.epochs, gpu=args.gpu, batch_size=args.batch_size, limit=args.limit):
                run_war_games(episodes=args.episodes)
                run_daemon()
                
    except KeyboardInterrupt:
        logger.warning("🛑 INTERRUPTED BY OPERATOR. Orderly shutdown initiated.")
        sys.exit(0)
    except Exception as e:
        logger.critical("💥 CRITICAL SYSTEM FAULT: %s", e)
        sys.exit(1)
