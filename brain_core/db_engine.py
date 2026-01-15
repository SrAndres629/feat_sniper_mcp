"""
db_engine.py - SQLite Database Engine for Unified Model
Persistent storage for state history, transitions, and calibrations.

Features:
- State history logging
- Transition tracking
- Calibration versioning
- Query interface for analysis
- Export capabilities
"""

import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from contextlib import contextmanager


@dataclass
class StateRecord:
    """Record of a single state observation."""
    timestamp: datetime
    symbol: str
    timeframe: str
    state: str
    confidence: float
    effort: float
    result: float
    compression: float
    slope: float
    speed: float
    feat_score: float


@dataclass
class TransitionRecord:
    """Record of a state transition."""
    timestamp: datetime
    symbol: str
    timeframe: str
    from_state: str
    to_state: str
    confidence: float
    reason: str


@dataclass
class CalibrationRecord:
    """Record of a calibration run."""
    timestamp: datetime
    symbol: str
    timeframe: str
    thresholds_json: str
    score: float
    method: str


class UnifiedModelDB:
    """
    SQLite database for Unified Model state history and calibration.
    
    Tables:
    - state_history: All state observations
    - transitions: State transition events
    - calibrations: Calibration runs and parameters
    """
    
    def __init__(self, db_path: str = "unified_model.db"):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self._connect()
        self._create_tables()
    
    def _connect(self) -> None:
        """Establish database connection."""
        self.conn = sqlite3.connect(
            self.db_path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        )
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
    
    @contextmanager
    def _transaction(self):
        """Context manager for transactions."""
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
    
    def _create_tables(self) -> None:
        """Create database schema."""
        with self._transaction():
            self.conn.executescript('''
                -- State history table
                CREATE TABLE IF NOT EXISTS state_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    state TEXT NOT NULL,
                    confidence REAL,
                    effort REAL,
                    result REAL,
                    compression REAL,
                    slope REAL,
                    speed REAL,
                    feat_score REAL
                );
                
                -- Transitions table
                CREATE TABLE IF NOT EXISTS transitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    from_state TEXT NOT NULL,
                    to_state TEXT NOT NULL,
                    confidence REAL,
                    reason TEXT
                );
                
                -- Calibrations table
                CREATE TABLE IF NOT EXISTS calibrations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    thresholds_json TEXT NOT NULL,
                    score REAL,
                    method TEXT,
                    is_active INTEGER DEFAULT 0
                );
                
                -- Indexes for common queries
                CREATE INDEX IF NOT EXISTS idx_state_symbol_tf_ts 
                    ON state_history(symbol, timeframe, timestamp);
                CREATE INDEX IF NOT EXISTS idx_trans_symbol_tf_ts 
                    ON transitions(symbol, timeframe, timestamp);
                CREATE INDEX IF NOT EXISTS idx_calib_symbol_tf 
                    ON calibrations(symbol, timeframe, is_active);
            ''')
    
    # ==========================================================================
    # State History Operations
    # ==========================================================================
    
    def log_state(self,
                  symbol: str,
                  timeframe: str,
                  state: str,
                  confidence: float,
                  metrics: Dict[str, float]) -> int:
        """
        Log current state to database.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Timeframe (e.g., 'H1')
            state: Current state name
            confidence: Confidence score (0-100)
            metrics: Dictionary with effort, result, compression, slope, speed, feat_score
        
        Returns:
            ID of inserted record
        """
        with self._transaction():
            cursor = self.conn.execute('''
                INSERT INTO state_history 
                (symbol, timeframe, state, confidence, effort, result, 
                 compression, slope, speed, feat_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol, timeframe, state, confidence,
                metrics.get('effort', 0),
                metrics.get('result', 0),
                metrics.get('compression', 0),
                metrics.get('slope', 0),
                metrics.get('speed', 0),
                metrics.get('feat_score', 0)
            ))
            return cursor.lastrowid
    
    def log_state_batch(self, records: List[StateRecord]) -> int:
        """Log multiple state records efficiently."""
        with self._transaction():
            cursor = self.conn.executemany('''
                INSERT INTO state_history 
                (timestamp, symbol, timeframe, state, confidence, effort, result, 
                 compression, slope, speed, feat_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', [
                (r.timestamp, r.symbol, r.timeframe, r.state, r.confidence,
                 r.effort, r.result, r.compression, r.slope, r.speed, r.feat_score)
                for r in records
            ])
            return len(records)
    
    def get_state_history(self,
                          symbol: str,
                          timeframe: str,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          limit: int = 1000) -> List[Dict]:
        """Get state history for a symbol/timeframe."""
        query = '''
            SELECT * FROM state_history
            WHERE symbol = ? AND timeframe = ?
        '''
        params: List[Any] = [symbol, timeframe]
        
        if start_time:
            query += ' AND timestamp >= ?'
            params.append(start_time)
        if end_time:
            query += ' AND timestamp <= ?'
            params.append(end_time)
        
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        cursor = self.conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    def get_state_distribution(self,
                                symbol: str,
                                timeframe: str,
                                days: int = 30) -> Dict[str, int]:
        """Get distribution of states over time period."""
        cursor = self.conn.execute('''
            SELECT state, COUNT(*) as count
            FROM state_history
            WHERE symbol = ? AND timeframe = ?
              AND timestamp > datetime('now', ?)
            GROUP BY state
            ORDER BY count DESC
        ''', (symbol, timeframe, f'-{days} days'))
        
        return {row['state']: row['count'] for row in cursor.fetchall()}
    
    # ==========================================================================
    # Transition Operations
    # ==========================================================================
    
    def log_transition(self,
                       symbol: str,
                       timeframe: str,
                       from_state: str,
                       to_state: str,
                       confidence: float,
                       reason: str = "") -> int:
        """Log a state transition."""
        with self._transaction():
            cursor = self.conn.execute('''
                INSERT INTO transitions 
                (symbol, timeframe, from_state, to_state, confidence, reason)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (symbol, timeframe, from_state, to_state, confidence, reason))
            return cursor.lastrowid
    
    def get_transition_stats(self,
                              symbol: str,
                              timeframe: str,
                              days: int = 30) -> List[Dict]:
        """Get transition statistics for analysis."""
        cursor = self.conn.execute('''
            SELECT 
                from_state, 
                to_state, 
                COUNT(*) as count,
                AVG(confidence) as avg_confidence,
                MIN(confidence) as min_confidence,
                MAX(confidence) as max_confidence
            FROM transitions
            WHERE symbol = ? AND timeframe = ?
              AND timestamp > datetime('now', ?)
            GROUP BY from_state, to_state
            ORDER BY count DESC
        ''', (symbol, timeframe, f'-{days} days'))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_transition_matrix(self,
                               symbol: str,
                               timeframe: str,
                               days: int = 30) -> Tuple[List[str], List[List[int]]]:
        """Get transition count matrix."""
        states = ['ACCUMULATION', 'EXPANSION', 'DISTRIBUTION', 'RESET']
        matrix = [[0] * 4 for _ in range(4)]
        
        stats = self.get_transition_stats(symbol, timeframe, days)
        state_to_idx = {s: i for i, s in enumerate(states)}
        
        for row in stats:
            from_idx = state_to_idx.get(row['from_state'], -1)
            to_idx = state_to_idx.get(row['to_state'], -1)
            if from_idx >= 0 and to_idx >= 0:
                matrix[from_idx][to_idx] = row['count']
        
        return states, matrix
    
    # ==========================================================================
    # Calibration Operations
    # ==========================================================================
    
    def save_calibration(self,
                         symbol: str,
                         timeframe: str,
                         thresholds: Dict,
                         score: float,
                         method: str,
                         set_active: bool = True) -> int:
        """Save a calibration configuration."""
        with self._transaction():
            # Deactivate previous calibrations if setting as active
            if set_active:
                self.conn.execute('''
                    UPDATE calibrations 
                    SET is_active = 0
                    WHERE symbol = ? AND timeframe = ?
                ''', (symbol, timeframe))
            
            cursor = self.conn.execute('''
                INSERT INTO calibrations 
                (symbol, timeframe, thresholds_json, score, method, is_active)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                symbol, timeframe, 
                json.dumps(thresholds),
                score, method,
                1 if set_active else 0
            ))
            return cursor.lastrowid
    
    def get_active_calibration(self,
                                symbol: str,
                                timeframe: str) -> Optional[Dict]:
        """Get the currently active calibration."""
        cursor = self.conn.execute('''
            SELECT * FROM calibrations
            WHERE symbol = ? AND timeframe = ? AND is_active = 1
            ORDER BY timestamp DESC
            LIMIT 1
        ''', (symbol, timeframe))
        
        row = cursor.fetchone()
        if row:
            result = dict(row)
            result['thresholds'] = json.loads(result['thresholds_json'])
            return result
        return None
    
    def get_calibration_history(self,
                                 symbol: str,
                                 timeframe: str,
                                 limit: int = 10) -> List[Dict]:
        """Get calibration history."""
        cursor = self.conn.execute('''
            SELECT * FROM calibrations
            WHERE symbol = ? AND timeframe = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (symbol, timeframe, limit))
        
        results = []
        for row in cursor.fetchall():
            result = dict(row)
            result['thresholds'] = json.loads(result['thresholds_json'])
            results.append(result)
        return results
    
    # ==========================================================================
    # Analysis Queries
    # ==========================================================================
    
    def get_performance_summary(self,
                                 symbol: str,
                                 timeframe: str,
                                 days: int = 30) -> Dict:
        """Get performance summary for a symbol/timeframe."""
        # State distribution
        state_dist = self.get_state_distribution(symbol, timeframe, days)
        
        # Average confidence by state
        cursor = self.conn.execute('''
            SELECT state, AVG(confidence) as avg_conf
            FROM state_history
            WHERE symbol = ? AND timeframe = ?
              AND timestamp > datetime('now', ?)
            GROUP BY state
        ''', (symbol, timeframe, f'-{days} days'))
        conf_by_state = {row['state']: row['avg_conf'] for row in cursor.fetchall()}
        
        # Transition count
        cursor = self.conn.execute('''
            SELECT COUNT(*) as count
            FROM transitions
            WHERE symbol = ? AND timeframe = ?
              AND timestamp > datetime('now', ?)
        ''', (symbol, timeframe, f'-{days} days'))
        trans_count = cursor.fetchone()['count']
        
        # Active calibration
        active_cal = self.get_active_calibration(symbol, timeframe)
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'period_days': days,
            'state_distribution': state_dist,
            'avg_confidence_by_state': conf_by_state,
            'total_transitions': trans_count,
            'active_calibration': active_cal['method'] if active_cal else None,
            'calibration_score': active_cal['score'] if active_cal else None
        }
    
    def export_to_csv(self,
                       table: str,
                       filepath: str,
                       symbol: Optional[str] = None,
                       timeframe: Optional[str] = None) -> int:
        """Export table data to CSV."""
        query = f'SELECT * FROM {table}'
        params: List[Any] = []
        
        conditions = []
        if symbol:
            conditions.append('symbol = ?')
            params.append(symbol)
        if timeframe:
            conditions.append('timeframe = ?')
            params.append(timeframe)
        
        if conditions:
            query += ' WHERE ' + ' AND '.join(conditions)
        
        cursor = self.conn.execute(query, params)
        rows = cursor.fetchall()
        
        if not rows:
            return 0
        
        import csv
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows([dict(row) for row in rows])
        
        return len(rows)
    
    # ==========================================================================
    # Maintenance
    # ==========================================================================
    
    def cleanup_old_data(self, days_to_keep: int = 90) -> Dict[str, int]:
        """Remove data older than specified days."""
        with self._transaction():
            result = {}
            
            cursor = self.conn.execute('''
                DELETE FROM state_history
                WHERE timestamp < datetime('now', ?)
            ''', (f'-{days_to_keep} days',))
            result['state_history'] = cursor.rowcount
            
            cursor = self.conn.execute('''
                DELETE FROM transitions
                WHERE timestamp < datetime('now', ?)
            ''', (f'-{days_to_keep} days',))
            result['transitions'] = cursor.rowcount
            
            # Keep all calibrations for audit trail
            result['calibrations'] = 0
            
            # Vacuum to reclaim space
            self.conn.execute('VACUUM')
            
            return result
    
    def get_table_stats(self) -> Dict[str, int]:
        """Get row counts for all tables."""
        tables = ['state_history', 'transitions', 'calibrations']
        stats = {}
        
        for table in tables:
            cursor = self.conn.execute(f'SELECT COUNT(*) as count FROM {table}')
            stats[table] = cursor.fetchone()['count']
        
        return stats

    def get_narrative_history(self, days: int = 30) -> List[str]:
        """
        Extrae la historia del modelo y la convierte en una narrativa natural para RAG.
        Combina estados y transiciones en una serie cronolgica legible.
        """
        narratives = []
        
        # 1. Obtener Transiciones (hitos clave)
        cursor = self.conn.execute('''
            SELECT timestamp, symbol, timeframe, from_state, to_state, confidence, reason
            FROM transitions
            WHERE timestamp > datetime('now', ?)
            ORDER BY timestamp ASC
        ''', (f'-{days} days',))
        
        for row in cursor.fetchall():
            ts_val = row['timestamp']
            if isinstance(ts_val, str):
                try:
                    ts_dt = datetime.fromisoformat(ts_val)
                except ValueError:
                    ts_dt = datetime.strptime(ts_val, "%Y-%m-%d %H:%M:%S")
                ts = ts_dt.strftime("%Y-%m-%d %H:%M")
            else:
                ts = ts_val.strftime("%Y-%m-%d %H:%M")

            narrative = (
                f"El {ts}, en {row['symbol']} ({row['timeframe']}), el mercado cambi de "
                f"{row['from_state']} a {row['to_state']} con una confianza del {row['confidence']:.1f}%. "
            )
            if row['reason']:
                narrative += f"Motivo tctico: {row['reason']}."
            narratives.append(narrative)

        # 2. Obtener Estados representativos (cada cierto tiempo o cambios de metrics)
        # Para evitar spam de memoria, tomamos los estados donde el feat_score fue extremo o hubo cambios
        cursor = self.conn.execute('''
            SELECT timestamp, symbol, timeframe, state, confidence, feat_score, compression, speed
            FROM state_history
            WHERE timestamp > datetime('now', ?)
              AND (feat_score > 80 OR feat_score < 20 OR id % 50 = 0)
            ORDER BY timestamp ASC
        ''', (f'-{days} days',))
        
        for row in cursor.fetchall():
            ts_val = row['timestamp']
            if isinstance(ts_val, str):
                try:
                    ts_dt = datetime.fromisoformat(ts_val)
                except ValueError:
                    ts_dt = datetime.strptime(ts_val, "%Y-%m-%d %H:%M:%S")
                ts = ts_dt.strftime("%Y-%m-%d %H:%M")
            else:
                ts = ts_val.strftime("%Y-%m-%d %H:%M")

            narrative = (
                f"Observacin en {ts}: {row['symbol']} en {row['state']}. "
                f"Score FEAT: {row['feat_score']:.1f}, Compresin: {row['compression']:.2f}, "
                f"Velocidad: {row['speed']:.2f}. Confianza del modelo: {row['confidence']:.1f}%."
            )
            narratives.append(narrative)
            
        return sorted(narratives) # Devolver en orden cronolgico
    
    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    print("DB Engine - Unified Model Database")
    print("="*60)
    
    # Create test database
    db_path = os.path.join(os.path.dirname(__file__), "test_unified_model.db")
    
    with UnifiedModelDB(db_path) as db:
        # Test state logging
        state_id = db.log_state(
            symbol='EURUSD',
            timeframe='H1',
            state='ACCUMULATION',
            confidence=75.5,
            metrics={
                'effort': 0.8,
                'result': 0.2,
                'compression': 0.75,
                'slope': 0.1,
                'speed': 0.05,
                'feat_score': 68.0
            }
        )
        print(f"[DBEngine] Logged state: ID={state_id}")
        
        # Test transition logging
        trans_id = db.log_transition(
            symbol='EURUSD',
            timeframe='H1',
            from_state='ACCUMULATION',
            to_state='EXPANSION',
            confidence=82.3,
            reason='Breakout confirmed'
        )
        print(f"[DBEngine] Logged transition: ID={trans_id}")
        
        # Test calibration saving
        calibration = {
            'accumulation_compression': 0.7,
            'expansion_slope': 0.3,
            'distribution_momentum': -0.2,
            'reset_speed': 2.0
        }
        cal_id = db.save_calibration(
            symbol='EURUSD',
            timeframe='H1',
            thresholds=calibration,
            score=85.5,
            method='optuna_tpe'
        )
        print(f"[DBEngine] Saved calibration: ID={cal_id}")
        
        # Get stats
        stats = db.get_table_stats()
        print(f"\n[DBEngine] Table stats: {stats}")
        
        # Get active calibration
        active = db.get_active_calibration('EURUSD', 'H1')
        print(f"[DBEngine] Active calibration: {active['method']} (score={active['score']})")
    
    print(f"\n[DBEngine] Test database created at: {db_path}")
