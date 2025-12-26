"""
run_pipeline.py - Master Orchestration Script for Unified Model
Executes the full data processing pipeline:
1. Ingest Data (CSV -> DB)
2. Train/Update Models (ML Engine)
3. Optimize Thresholds (Optuna)
4. visualizer Results (Viz Engine)
5. Export Configuration (Interop)

Usage:
    python run_pipeline.py --symbol EURUSD --timeframe H1 --data path/to/data.csv
"""

import os
import sys
import argparse
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional

# Import custom engines
from db_engine import UnifiedModelDB, StateRecord
from ml_engine import train_fsm_classifier, FSMClassifier
from optuna_optimizer import run_quick_optimization, OptunaOptimizer
from viz_engine import InstitutionalDashboard

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Pipeline")

class PipelineError(Exception):
    """Custom exception for pipeline failures."""
    pass

def load_and_validate_csv(filepath: str) -> pd.DataFrame:
    """
    Load data from CSV and validate required columns.
    
    Expected columns:
    - time (or timestamp)
    - effort (volume/normalized)
    - result (price change/atr)
    - compression (0-1)
    - slope (normalized)
    - speed (normalized)
    """
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        
        # Normalize column names
        df.columns = [c.lower().strip() for c in df.columns]
        
        required_cols = ['effort', 'result', 'compression', 'slope', 'speed']
        missing = [c for c in required_cols if c not in df.columns]
        
        if missing:
            raise PipelineError(f"Missing required columns in CSV: {missing}")
            
        # Fill NaNs
        df.fillna(method='ffill', inplace=True)
        df.fillna(0, inplace=True)
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading CSV: {str(e)}")
        raise PipelineError(f"Data loading failed: {str(e)}")

def ingest_to_db(db: UnifiedModelDB, df: pd.DataFrame, symbol: str, timeframe: str) -> None:
    """
    Ingest DataFrame into SQLite database.
    """
    try:
        logger.info(f"Ingesting {len(df)} records into DB for {symbol} {timeframe}")
        
        records = []
        for _, row in df.iterrows():
            # Create timestamp if missing or parse it
            ts = datetime.now() # Default fallback
            if 'time' in row:
                try:
                    ts = pd.to_datetime(row['time']).to_pydatetime()
                except:
                    pass
            elif 'timestamp' in row:
                try:
                    ts = pd.to_datetime(row['timestamp']).to_pydatetime()
                except:
                    pass
            
            # Simple heuristic state for initial logging (can be updated by ML later)
            # This is just for record keeping if state isn't in CSV
            state = row.get('state', 'UNKNOWN')
                
            record = StateRecord(
                timestamp=ts,
                symbol=symbol,
                timeframe=timeframe,
                state=state,
                confidence=row.get('confidence', 50.0),
                effort=row['effort'],
                result=row['result'],
                compression=row['compression'],
                slope=row['slope'],
                speed=row['speed'],
                feat_score=row.get('feat_score', 0.0)
            )
            records.append(record)
            
        db.log_state_batch(records)
        logger.info("Ingestion completed successfully.")
        
    except Exception as e:
        logger.error(f"Error during DB ingestion: {str(e)}")
        raise PipelineError(f"DB Ingestion failed: {str(e)}")

def run_ml_pipeline(df: pd.DataFrame, output_dir: str) -> FSMClassifier:
    """
    Train and export ML model.
    """
    try:
        logger.info("Starting ML training pipeline...")
        
        # Extract features as numpy arrays
        effort = df['effort'].values
        result = df['result'].values
        compression = df['compression'].values
        slope = df['slope'].values
        speed = df['speed'].values
        
        # Train Classifier
        classifier = train_fsm_classifier(
            effort, result, compression, slope, speed, model_type='rf'
        )
        
        # Save Model
        model_path = os.path.join(output_dir, "fsm_model.joblib")
        classifier.save(model_path)
        logger.info(f"ML Model saved to {model_path}")
        
        # Export for MQL5
        mql5_path = os.path.join(output_dir, "ml_thresholds.txt")
        classifier.export_for_mql5(mql5_path)
        logger.info(f"MQL5 ML params exported to {mql5_path}")
        
        return classifier
        
    except Exception as e:
        logger.error(f"Error in ML pipeline: {str(e)}")
        raise PipelineError(f"ML Pipeline failed: {str(e)}")

def run_optimization_pipeline(df: pd.DataFrame, output_dir: str, symbol: str, timeframe: str) -> Dict:
    """
    Run Optuna optimization for thresholds.
    """
    try:
        logger.info("Starting Bayesian Optimization pipeline...")
        
        effort = df['effort'].values
        result = df['result'].values
        compression = df['compression'].values
        slope = df['slope'].values
        speed = df['speed'].values
        
        # Run Quick Optimization (tweak n_trials for production)
        optimizer = OptunaOptimizer()
        optimizer.set_data(effort, result, compression, slope, speed)
        best_thresholds = optimizer.optimize()
        
        # Export Results
        calib_path = os.path.join(output_dir, "optuna_calibration.txt")
        optimizer.export_calibration(calib_path, symbol, timeframe)
        logger.info(f"Optimized thresholds exported to {calib_path}")
        
        # Generate generic plot if possible
        # optimizer.plot_optimization_history(os.path.join(output_dir, "opt_history.html"))
        
        return optimizer.get_results()
        
    except Exception as e:
        logger.error(f"Error in Optimization pipeline: {str(e)}")
        raise PipelineError(f"Optimization Pipeline failed: {str(e)}")

def run_viz_pipeline(df: pd.DataFrame, 
                     classifier: FSMClassifier, 
                     opt_results: Dict,
                     output_dir: str,
                     symbol: str, 
                     timeframe: str) -> None:
    """
    Generate comprehensive dashboard.
    """
    try:
        logger.info("Generating Visualization Dashboard...")
        
        # Prepare data for dashboard
        timestamps = df.get('time', pd.date_range(end=datetime.now(), periods=len(df), freq='H')).tolist()
        prices = df.get('close', np.cumsum(np.random.randn(len(df))) + 100).tolist() # Fallback
        
        # Get ML predictions for the whole dataset
        X = classifier.prepare_features(
            df['effort'].values, 
            df['result'].values, 
            df['compression'].values, 
            df['slope'].values, 
            df['speed'].values
        )
        preds, confidences = classifier.predict_with_confidence(X)
        states = [classifier.STATE_NAMES[p] for p in preds]
        
        dashboard = InstitutionalDashboard(title=f"Unified Model: {symbol} {timeframe}")
        
        # Fake FEAT scores for demo (in prod, load from CSV)
        feat_scores = {'form': 70, 'space': 60, 'accel': 80, 'time': 50}
        
        # Feature importances from ML model
        importances = classifier.training_metrics.get('feature_importances', {})
        
        fig = dashboard.create_full_dashboard(
            timestamps=timestamps,
            states=states,
            confidences=confidences.tolist(),
            prices=prices,
            feat_scores=feat_scores,
            feature_importances=importances,
            current_metrics={'Score': opt_results.get('best_value', 0)}
        )
        
        dash_path = os.path.join(output_dir, "dashboard.html")
        dashboard.save_dashboard(fig, dash_path)
        logger.info(f"Dashboard saved to {dash_path}")
        
    except Exception as e:
        logger.error(f"Error in Visualization pipeline: {str(e)}")
        raise PipelineError(f"Visualization Pipeline failed: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Unified Model Orchestration Pipeline")
    parser.add_argument("--symbol", type=str, default="EURUSD", help="Trading Symbol")
    parser.add_argument("--timeframe", type=str, default="H1", help="Timeframe")
    parser.add_argument("--data", type=str, required=False, help="Path to input CSV")
    parser.add_argument("--output", type=str, default=".", help="Output directory")
    parser.add_argument("--db", type=str, default="unified_model.db", help="Database file")
    
    args = parser.parse_args()
    
    try:
        # Generate mock data if no file provided (for testing)
        if not args.data:
            logger.warning("No data file provided. Generating synthetic data for testing.")
            args.data = "mock_data.csv"
            # create mock csv
            n = 1000
            mock_df = pd.DataFrame({
                'time': pd.date_range(end=datetime.now(), periods=n, freq='H'),
                'effort': np.random.random(n),
                'result': np.random.random(n),
                'compression': np.random.random(n),
                'slope': np.random.randn(n),
                'speed': np.random.random(n),
                'close': np.cumsum(np.random.randn(n)) + 1.1000,
                'confidence': np.random.uniform(50, 100, n),
                'feat_score': np.random.uniform(0, 100, n)
            })
            mock_df.to_csv(args.data, index=False)
            logger.info(f"Created mock data: {args.data}")

        # Ensure output dir exists
        os.makedirs(args.output, exist_ok=True)
        
        # 1. Load Data
        df = load_and_validate_csv(args.data)
        
        # 2. DB Ingestion
        db_path = os.path.join(args.output, args.db)
        with UnifiedModelDB(db_path) as db:
            ingest_to_db(db, df, args.symbol, args.timeframe)
            
        # 3. ML Pipeline
        classifier = run_ml_pipeline(df, args.output)
        
        # 4. Optimization Pipeline
        opt_results = run_optimization_pipeline(df, args.output, args.symbol, args.timeframe)
        
        # 5. Visualization Pipeline
        run_viz_pipeline(df, classifier, opt_results, args.output, args.symbol, args.timeframe)
        
        logger.info("Pipeline completed successfully! ✅")
        print("\n=== PIPELINE SUCCESS ===")
        print(f"1. Database: {db_path}")
        print(f"2. ML Model: {os.path.join(args.output, 'fsm_model.joblib')}")
        print(f"3. MQL5 Parameters: {os.path.join(args.output, 'optuna_calibration.txt')}")
        print(f"4. Dashboard: {os.path.join(args.output, 'dashboard.html')}")
        
    except PipelineError as e:
        logger.critical(f"Pipeline Terminated: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unexpected Fatal Error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
