"""
viz_engine.py - Institutional Dashboard Visualization
Plotly-based interactive dashboards for Unified Model analysis.

Features:
- State timeline with confidence overlay
- FEAT radar charts
- Transition heatmaps
- Feature importance visualization
- Export to standalone HTML
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import os

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("[VizEngine] plotly not installed. Install with: pip install plotly")


# Color scheme
COLORS = {
    'ACCUMULATION': '#3498db',
    'EXPANSION': '#2ecc71', 
    'DISTRIBUTION': '#f39c12',
    'RESET': '#9b59b6',
    'background': '#1a1a2e',
    'text': '#ffffff',
    'grid': '#333355',
    'accent': '#00d4ff'
}


class InstitutionalDashboard:
    """
    Plotly-based institutional dashboard for Unified Model visualization.
    """
    
    def __init__(self, title: str = "Unified Institutional Model"):
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly is required. Install with: pip install plotly")
        
        self.title = title
        self.template = 'plotly_dark'
    
    def create_state_timeline(self,
                              timestamps: List,
                              states: List[str],
                              confidences: List[float],
                              prices: Optional[List[float]] = None) -> go.Figure:
        """
        Create state timeline with confidence overlay.
        
        Args:
            timestamps: List of timestamps
            states: List of state names
            confidences: List of confidence values (0-100)
            prices: Optional list of prices
        
        Returns:
            Plotly Figure
        """
        n_rows = 3 if prices is not None else 2
        row_heights = [0.5, 0.25, 0.25] if prices is not None else [0.6, 0.4]
        
        fig = make_subplots(
            rows=n_rows,
            cols=1,
            shared_xaxes=True,
            row_heights=row_heights,
            vertical_spacing=0.05,
            subplot_titles=['Market State', 'Confidence', 'Price'] if prices else ['Market State', 'Confidence']
        )
        
        # State background colors
        state_numeric = [
            ['ACCUMULATION', 'EXPANSION', 'DISTRIBUTION', 'RESET'].index(s)
            for s in states
        ]
        
        # State timeline as scatter
        for state_name in COLORS:
            if state_name == 'background' or state_name == 'text':
                continue
            mask = [s == state_name for s in states]
            if not any(mask):
                continue
            
            fig.add_trace(
                go.Scatter(
                    x=[t for t, m in zip(timestamps, mask) if m],
                    y=[state_numeric[i] for i, m in enumerate(mask) if m],
                    mode='markers',
                    marker=dict(
                        color=COLORS.get(state_name, '#888'),
                        size=8,
                        symbol='square'
                    ),
                    name=state_name,
                    hovertemplate=f'{state_name}<br>%{{x}}<extra></extra>'
                ),
                row=1, col=1
            )
        
        fig.update_yaxes(
            ticktext=['ACCUM', 'EXP', 'DIST', 'RESET'],
            tickvals=[0, 1, 2, 3],
            row=1, col=1
        )
        
        # Confidence line
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=confidences,
                mode='lines',
                line=dict(color=COLORS['accent'], width=2),
                name='Confidence',
                hovertemplate='%{y:.1f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add 50% reference line
        fig.add_hline(
            y=50, 
            line_dash="dash", 
            line_color="gray",
            row=2, col=1
        )
        
        # Price if provided
        if prices is not None:
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=prices,
                    mode='lines',
                    line=dict(color='#ffffff', width=1),
                    name='Price',
                    hovertemplate='%{y:.5f}<extra></extra>'
                ),
                row=3, col=1
            )
        
        fig.update_layout(
            title=dict(text=self.title, font=dict(size=24)),
            template=self.template,
            height=600 if prices is None else 800,
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        
        return fig
    
    def create_feat_radar(self,
                          form_score: float,
                          space_score: float,
                          accel_score: float,
                          time_score: float,
                          title: str = "FEAT Analysis") -> go.Figure:
        """
        Create FEAT radar chart.
        
        Args:
            form_score: Form score (0-100)
            space_score: Space score (0-100)
            accel_score: Acceleration score (0-100)
            time_score: Time score (0-100)
            title: Chart title
        
        Returns:
            Plotly Figure
        """
        categories = ['Form (F)', 'Space (E)', 'Acceleration (A)', 'Time (T)']
        values = [form_score, space_score, accel_score, time_score]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # Close polygon
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(0, 212, 255, 0.3)',
            line=dict(color=COLORS['accent'], width=2),
            name='FEAT'
        ))
        
        # Add reference circle at 50%
        ref_values = [50, 50, 50, 50, 50]
        fig.add_trace(go.Scatterpolar(
            r=ref_values,
            theta=categories + [categories[0]],
            mode='lines',
            line=dict(color='gray', dash='dash', width=1),
            name='Reference'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    range=[0, 100],
                    showticklabels=True,
                    ticksuffix='%'
                ),
                angularaxis=dict(direction='clockwise')
            ),
            title=dict(text=title, font=dict(size=18)),
            template=self.template,
            showlegend=False,
            height=400
        )
        
        return fig
    
    def create_transition_heatmap(self,
                                   transition_matrix: np.ndarray,
                                   state_names: List[str] = None) -> go.Figure:
        """
        Create transition probability heatmap.
        
        Args:
            transition_matrix: N x N matrix of transition counts
            state_names: List of state names
        
        Returns:
            Plotly Figure
        """
        if state_names is None:
            state_names = ['ACCUM', 'EXP', 'DIST', 'RESET']
        
        # Normalize to percentages
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        normalized = (transition_matrix / row_sums) * 100
        
        fig = go.Figure(data=go.Heatmap(
            z=normalized,
            x=state_names,
            y=state_names,
            colorscale='Viridis',
            text=np.round(normalized, 1),
            texttemplate='%{text}%',
            textfont=dict(size=12),
            hoverongaps=False,
            hovertemplate='%{y} → %{x}: %{z:.1f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title='State Transition Probabilities',
            xaxis_title='To State',
            yaxis_title='From State',
            template=self.template,
            height=400
        )
        
        return fig
    
    def create_feature_importance(self,
                                   importances: Dict[str, float],
                                   title: str = "Feature Importances") -> go.Figure:
        """
        Create feature importance bar chart.
        
        Args:
            importances: Dictionary of feature name to importance score
            title: Chart title
        
        Returns:
            Plotly Figure
        """
        # Sort by importance
        sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        features = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]
        
        # Color gradient
        colors = [f'rgba(0, {int(180 + 75*v)}, {int(200 + 55*v)}, 0.8)' 
                  for v in np.linspace(1, 0, len(values))]
        
        fig = go.Figure(data=go.Bar(
            x=values,
            y=features,
            orientation='h',
            marker=dict(color=colors),
            text=[f'{v:.3f}' for v in values],
            textposition='auto',
            hovertemplate='%{y}: %{x:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Importance',
            yaxis_title='Feature',
            template=self.template,
            height=max(300, len(features) * 40)
        )
        
        return fig
    
    def create_metrics_panel(self,
                              metrics: Dict[str, float],
                              title: str = "Current Metrics") -> go.Figure:
        """
        Create metrics indicator panel.
        
        Args:
            metrics: Dictionary of metric names to values
            title: Panel title
        
        Returns:
            Plotly Figure
        """
        n_metrics = len(metrics)
        cols = min(4, n_metrics)
        rows = (n_metrics + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows,
            cols=cols,
            specs=[[{'type': 'indicator'} for _ in range(cols)] for _ in range(rows)],
            horizontal_spacing=0.1,
            vertical_spacing=0.2
        )
        
        for i, (name, value) in enumerate(metrics.items()):
            row = i // cols + 1
            col = i % cols + 1
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=value,
                    title=dict(text=name, font=dict(size=12)),
                    gauge=dict(
                        axis=dict(range=[0, 100]),
                        bar=dict(color=COLORS['accent']),
                        bgcolor='#333355',
                        borderwidth=2,
                        bordercolor='#555'
                    )
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title=title,
            template=self.template,
            height=200 * rows
        )
        
        return fig
    
    def create_full_dashboard(self,
                               timestamps: List,
                               states: List[str],
                               confidences: List[float],
                               prices: List[float],
                               feat_scores: Dict[str, float],
                               feature_importances: Dict[str, float],
                               current_metrics: Dict[str, float]) -> go.Figure:
        """
        Create full dashboard with all components.
        
        Returns:
            Plotly Figure with all panels
        """
        fig = make_subplots(
            rows=3, cols=2,
            row_heights=[0.5, 0.25, 0.25],
            column_widths=[0.7, 0.3],
            specs=[
                [{'colspan': 2}, None],
                [{'type': 'xy'}, {'type': 'polar'}],
                [{'type': 'xy'}, {'type': 'xy'}]
            ],
            subplot_titles=[
                'State Timeline & Price',
                'Confidence', 'FEAT Analysis',
                'Feature Importance', ''
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # Main timeline (row 1, spans both cols)
        state_numeric = [
            ['ACCUMULATION', 'EXPANSION', 'DISTRIBUTION', 'RESET'].index(s)
            for s in states
        ]
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=prices,
                mode='lines',
                line=dict(color='white', width=1),
                name='Price'
            ),
            row=1, col=1
        )
        
        # Confidence (row 2, col 1)
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=confidences,
                mode='lines',
                fill='tozeroy',
                line=dict(color=COLORS['accent'], width=2),
                name='Confidence'
            ),
            row=2, col=1
        )
        
        # FEAT radar (row 2, col 2)
        categories = ['Form', 'Space', 'Accel', 'Time']
        radar_values = [
            feat_scores.get('form', 50),
            feat_scores.get('space', 50),
            feat_scores.get('accel', 50),
            feat_scores.get('time', 50)
        ]
        
        fig.add_trace(
            go.Scatterpolar(
                r=radar_values + [radar_values[0]],
                theta=categories + [categories[0]],
                fill='toself',
                fillcolor='rgba(0, 212, 255, 0.3)',
                line=dict(color=COLORS['accent'], width=2),
                name='FEAT'
            ),
            row=2, col=2
        )
        
        # Feature importance (row 3, col 1)
        sorted_features = sorted(feature_importances.items(), 
                                  key=lambda x: x[1], reverse=True)[:6]
        fig.add_trace(
            go.Bar(
                x=[v for _, v in sorted_features],
                y=[k for k, _ in sorted_features],
                orientation='h',
                marker=dict(color=COLORS['accent']),
                name='Importance'
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            title=dict(
                text=f'<b>{self.title}</b>',
                font=dict(size=24, color='white')
            ),
            template=self.template,
            height=900,
            showlegend=True,
            legend=dict(orientation='h', y=1.02)
        )
        
        return fig
    
    def save_dashboard(self, fig: go.Figure, filepath: str) -> None:
        """Save dashboard as standalone HTML."""
        fig.write_html(
            filepath,
            include_plotlyjs=True,
            full_html=True,
            config={'displayModeBar': True, 'responsive': True}
        )
        print(f"[VizEngine] Saved dashboard to {filepath}")


def create_sample_dashboard(output_path: str) -> None:
    """Create sample dashboard with synthetic data."""
    if not PLOTLY_AVAILABLE:
        print("[VizEngine] plotly not installed")
        return
    
    # Generate synthetic data
    np.random.seed(42)
    n = 200
    
    timestamps = [datetime(2024, 1, 1, i // 60, i % 60) for i in range(n)]
    states = np.random.choice(
        ['ACCUMULATION', 'EXPANSION', 'DISTRIBUTION', 'RESET'],
        n, p=[0.4, 0.3, 0.2, 0.1]
    ).tolist()
    confidences = np.clip(np.random.normal(65, 15, n), 20, 95).tolist()
    prices = np.cumsum(np.random.randn(n) * 0.0005) + 1.1000
    
    # Create dashboard
    dashboard = InstitutionalDashboard()
    
    fig = dashboard.create_state_timeline(
        timestamps, states, confidences, prices.tolist()
    )
    
    dashboard.save_dashboard(fig, output_path)


if __name__ == "__main__":
    print("Viz Engine - Institutional Dashboard")
    print("="*60)
    
    if not PLOTLY_AVAILABLE:
        print("Please install plotly: pip install plotly")
        exit(1)
    
    # Create sample dashboard
    output_path = os.path.join(os.path.dirname(__file__), "dashboard_sample.html")
    create_sample_dashboard(output_path)
    
    # Create individual charts
    dashboard = InstitutionalDashboard()
    
    # FEAT radar
    feat_fig = dashboard.create_feat_radar(75, 60, 80, 55)
    feat_fig.write_html(
        os.path.join(os.path.dirname(__file__), "feat_radar.html")
    )
    
    # Feature importance
    importances = {
        'effort_pct': 0.25,
        'result_pct': 0.22,
        'compression': 0.18,
        'slope': 0.15,
        'speed': 0.12,
        'momentum': 0.08
    }
    imp_fig = dashboard.create_feature_importance(importances)
    imp_fig.write_html(
        os.path.join(os.path.dirname(__file__), "feature_importance.html")
    )
    
    print("[VizEngine] Sample dashboards created")
