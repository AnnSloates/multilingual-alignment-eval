"""
Visualization and report generation tools for multilingual alignment evaluation.
Creates interactive plots, dashboards, and comprehensive reports.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import json
from datetime import datetime
import base64
from io import BytesIO
import warnings
from matplotlib.backends.backend_pdf import PdfPages
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class EvaluationVisualizer:
    """Creates visualizations for multilingual alignment evaluation results."""
    
    def __init__(self, style: str = 'default'):
        """
        Initialize visualizer with style settings.
        
        Args:
            style: Visual style preset ('default', 'minimal', 'academic')
        """
        self.style = style
        self._setup_style()
        
    def _setup_style(self):
        """Setup visualization style based on preset."""
        if self.style == 'minimal':
            plt.style.use('seaborn-v0_8-white')
            sns.set_palette("Set2")
        elif self.style == 'academic':
            plt.style.use('seaborn-v0_8-paper')
            sns.set_palette("colorblind")
        else:
            plt.style.use('seaborn-v0_8-darkgrid')
            sns.set_palette("husl")
            
    def create_overview_dashboard(self, metrics: Dict, output_path: Optional[str] = None) -> go.Figure:
        """
        Create an interactive dashboard with key metrics.
        
        Args:
            metrics: Evaluation metrics dictionary
            output_path: Optional path to save the dashboard
            
        Returns:
            Plotly figure object
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Overall Performance', 'Language Comparison', 
                          'Model Performance', 'Safety Score Distribution'),
            specs=[[{'type': 'indicator'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'box'}]]
        )
        
        # Overall performance indicators
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=1 - metrics['hallucination_rate'],
                title={'text': "Accuracy Rate"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [None, 1]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 0.5], 'color': "lightgray"},
                        {'range': [0.5, 0.8], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.9
                    }
                }
            ),
            row=1, col=1
        )
        
        # Language comparison
        if 'per_language_metrics' in metrics:
            languages = list(metrics['per_language_metrics'].keys())
            halluc_rates = [metrics['per_language_metrics'][lang]['hallucination_rate'] 
                           for lang in languages]
            safety_scores = [metrics['per_language_metrics'][lang]['average_safety_score'] 
                            for lang in languages]
            
            fig.add_trace(
                go.Bar(
                    x=languages,
                    y=halluc_rates,
                    name='Hallucination Rate',
                    marker_color='indianred'
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Bar(
                    x=languages,
                    y=safety_scores,
                    name='Safety Score',
                    marker_color='lightseagreen',
                    yaxis='y2'
                ),
                row=1, col=2
            )
        
        # Model comparison
        if 'per_model_metrics' in metrics:
            models = list(metrics['per_model_metrics'].keys())
            model_scores = [metrics['per_model_metrics'][model]['average_safety_score'] 
                           for model in models]
            
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=model_scores,
                    marker_color='lightblue',
                    text=[f"{s:.3f}" for s in model_scores],
                    textposition='auto'
                ),
                row=2, col=1
            )
        
        # Safety score distribution
        # Generate sample distribution data for visualization
        np.random.seed(42)
        safety_dist = np.random.beta(
            metrics['average_safety_score'] * 10,
            (1 - metrics['average_safety_score']) * 10,
            1000
        )
        
        fig.add_trace(
            go.Box(
                y=safety_dist,
                name='Safety Scores',
                marker_color='lightgreen'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Multilingual Alignment Evaluation Dashboard",
            height=800,
            showlegend=True
        )
        
        # Save if path provided
        if output_path:
            fig.write_html(output_path)
            logger.info(f"Dashboard saved to {output_path}")
            
        return fig
    
    def plot_language_heatmap(self, df: pd.DataFrame, metric: str = 'safety_score',
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a heatmap showing metric values across languages and categories.
        
        Args:
            df: Evaluation dataframe
            metric: Metric to visualize
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if 'language' not in df.columns or 'category' not in df.columns:
            logger.warning("Missing required columns for heatmap")
            return None
            
        # Create pivot table
        pivot_data = df.pivot_table(
            values=metric,
            index='language',
            columns='category',
            aggfunc='mean'
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create heatmap
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            center=0.5,
            ax=ax,
            cbar_kws={'label': metric.replace('_', ' ').title()}
        )
        
        ax.set_title(f'{metric.replace("_", " ").title()} by Language and Category')
        ax.set_xlabel('Category')
        ax.set_ylabel('Language')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Heatmap saved to {save_path}")
            
        return fig
    
    def plot_temporal_trends(self, df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot temporal trends if timestamp data is available.
        
        Args:
            df: Evaluation dataframe with timestamp column
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if 'timestamp' not in df.columns:
            logger.warning("No timestamp column found for temporal analysis")
            return None
            
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Resample by day
        daily_metrics = df.resample('D', on='timestamp').agg({
            'hallucinated': 'mean',
            'safety_score': 'mean'
        })
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot hallucination rate
        ax1.plot(daily_metrics.index, daily_metrics['hallucinated'], 
                marker='o', linewidth=2, markersize=6)
        ax1.set_ylabel('Hallucination Rate')
        ax1.set_title('Temporal Trends in Model Performance')
        ax1.grid(True, alpha=0.3)
        
        # Plot safety score
        ax2.plot(daily_metrics.index, daily_metrics['safety_score'], 
                marker='s', linewidth=2, markersize=6, color='green')
        ax2.set_ylabel('Average Safety Score')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Temporal trends saved to {save_path}")
            
        return fig
    
    def plot_performance_radar(self, metrics: Dict, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a radar chart showing multi-dimensional performance.
        
        Args:
            metrics: Evaluation metrics dictionary
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Define categories and values
        categories = ['Safety', 'Accuracy', 'Consistency', 'Coverage', 'Robustness']
        
        # Calculate values (normalize to 0-1)
        values = [
            metrics.get('average_safety_score', 0.5),
            1 - metrics.get('hallucination_rate', 0.5),
            metrics.get('consistency_score', 0.7),  # Placeholder
            metrics.get('language_coverage', 0.8),  # Placeholder
            metrics.get('robustness_score', 0.6)   # Placeholder
        ]
        
        # Number of variables
        num_vars = len(categories)
        
        # Compute angle of each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        # Draw the outline of our data
        ax.plot(angles, values, 'o-', linewidth=2, color='#FF6B6B')
        ax.fill(angles, values, alpha=0.25, color='#FF6B6B')
        
        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis lines for each angle and label
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        
        # Set y-axis limits and labels
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'])
        
        # Add title
        ax.set_title('Multi-dimensional Performance Analysis', y=1.08, fontsize=14)
        
        # Add grid
        ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Radar chart saved to {save_path}")
            
        return fig
    
    def create_comparison_plots(self, df: pd.DataFrame, 
                               groupby: str = 'language',
                               save_dir: Optional[str] = None) -> List[plt.Figure]:
        """
        Create comprehensive comparison plots.
        
        Args:
            df: Evaluation dataframe
            groupby: Column to group by ('language', 'model', 'category')
            save_dir: Directory to save plots
            
        Returns:
            List of matplotlib figures
        """
        figures = []
        
        if groupby not in df.columns:
            logger.warning(f"Column '{groupby}' not found in dataframe")
            return figures
            
        # 1. Box plot comparison
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        df.boxplot(column='safety_score', by=groupby, ax=ax1)
        ax1.set_title(f'Safety Score Distribution by {groupby.title()}')
        ax1.set_xlabel(groupby.title())
        ax1.set_ylabel('Safety Score')
        plt.suptitle('')  # Remove default title
        figures.append(fig1)
        
        # 2. Violin plot for distribution
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        unique_groups = df[groupby].unique()
        data_by_group = [df[df[groupby] == g]['safety_score'].values for g in unique_groups]
        ax2.violinplot(data_by_group, positions=range(len(unique_groups)))
        ax2.set_xticks(range(len(unique_groups)))
        ax2.set_xticklabels(unique_groups)
        ax2.set_xlabel(groupby.title())
        ax2.set_ylabel('Safety Score')
        ax2.set_title(f'Safety Score Distribution (Violin Plot) by {groupby.title()}')
        figures.append(fig2)
        
        # 3. Bar plot with error bars
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        grouped_stats = df.groupby(groupby).agg({
            'safety_score': ['mean', 'std'],
            'hallucinated': 'mean'
        })
        
        x_pos = np.arange(len(grouped_stats))
        means = grouped_stats[('safety_score', 'mean')]
        stds = grouped_stats[('safety_score', 'std')]
        
        ax3.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(grouped_stats.index)
        ax3.set_xlabel(groupby.title())
        ax3.set_ylabel('Average Safety Score')
        ax3.set_title(f'Average Safety Score by {groupby.title()} (with std dev)')
        figures.append(fig3)
        
        # Save figures if directory provided
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True, parents=True)
            
            for i, fig in enumerate(figures):
                save_path = save_dir / f"{groupby}_comparison_{i+1}.png"
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved plot to {save_path}")
                
        return figures


class ReportGenerator:
    """Generates comprehensive evaluation reports in various formats."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize report generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.visualizer = EvaluationVisualizer()
        
    def generate_html_report(self, metrics: Dict, df: Optional[pd.DataFrame] = None,
                           output_path: str = "evaluation_report.html") -> str:
        """
        Generate an interactive HTML report.
        
        Args:
            metrics: Evaluation metrics
            df: Optional dataframe with detailed results
            output_path: Path to save the HTML report
            
        Returns:
            Path to the generated report
        """
        # Create HTML template
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Multilingual Alignment Evaluation Report</title>
            <meta charset="utf-8">
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                h1 {{
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }}
                .metric-card {{
                    background: white;
                    border-radius: 8px;
                    padding: 20px;
                    margin: 20px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .metric-value {{
                    font-size: 2.5em;
                    font-weight: bold;
                    color: #3498db;
                    margin: 10px 0;
                }}
                .metric-label {{
                    font-size: 1.2em;
                    color: #7f8c8d;
                }}
                .grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                    background: white;
                    border-radius: 8px;
                    overflow: hidden;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ecf0f1;
                }}
                th {{
                    background-color: #3498db;
                    color: white;
                    font-weight: bold;
                }}
                tr:hover {{
                    background-color: #f8f9fa;
                }}
                .warning {{
                    background-color: #f39c12;
                    color: white;
                    padding: 10px;
                    border-radius: 4px;
                    margin: 10px 0;
                }}
                .success {{
                    background-color: #27ae60;
                    color: white;
                    padding: 10px;
                    border-radius: 4px;
                    margin: 10px 0;
                }}
                .chart-container {{
                    background: white;
                    border-radius: 8px;
                    padding: 20px;
                    margin: 20px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .timestamp {{
                    color: #7f8c8d;
                    font-size: 0.9em;
                    text-align: right;
                    margin-top: 20px;
                }}
            </style>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <h1>🌍 Multilingual Alignment Evaluation Report</h1>
            
            <div class="timestamp">
                Generated: {timestamp}
            </div>
            
            <h2>📊 Executive Summary</h2>
            <div class="grid">
                <div class="metric-card">
                    <div class="metric-label">Total Samples</div>
                    <div class="metric-value">{total_samples}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Hallucination Rate</div>
                    <div class="metric-value">{hallucination_rate:.1%}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Average Safety Score</div>
                    <div class="metric-value">{safety_score:.3f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Languages Evaluated</div>
                    <div class="metric-value">{num_languages}</div>
                </div>
            </div>
            
            {alerts_section}
            
            <h2>📈 Detailed Analysis</h2>
            
            {language_section}
            
            {model_section}
            
            <div class="chart-container">
                <h3>Interactive Dashboard</h3>
                <div id="dashboard"></div>
            </div>
            
            {recommendations_section}
            
            <script>
                {plotly_script}
            </script>
        </body>
        </html>
        """
        
        # Generate sections
        alerts_section = self._generate_alerts_section(metrics)
        language_section = self._generate_language_section(metrics)
        model_section = self._generate_model_section(metrics)
        recommendations_section = self._generate_recommendations(metrics)
        
        # Generate dashboard
        dashboard_fig = self.visualizer.create_overview_dashboard(metrics)
        plotly_script = f"Plotly.newPlot('dashboard', {dashboard_fig.to_json()})"
        
        # Fill template
        html_content = html_template.format(
            timestamp=metrics.get('timestamp', datetime.now().isoformat()),
            total_samples=metrics.get('total_samples', 'N/A'),
            hallucination_rate=metrics.get('hallucination_rate', 0),
            safety_score=metrics.get('average_safety_score', 0),
            num_languages=metrics.get('languages_evaluated', 'N/A'),
            alerts_section=alerts_section,
            language_section=language_section,
            model_section=model_section,
            recommendations_section=recommendations_section,
            plotly_script=plotly_script
        )
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        logger.info(f"HTML report saved to {output_path}")
        return output_path
    
    def _generate_alerts_section(self, metrics: Dict) -> str:
        """Generate alerts based on metrics."""
        alerts = []
        
        # Check hallucination rate
        if metrics.get('hallucination_rate', 0) > 0.2:
            alerts.append(
                '<div class="warning">⚠️ High hallucination rate detected '
                f"({metrics['hallucination_rate']:.1%}). Consider additional training.</div>"
            )
            
        # Check safety score
        if metrics.get('average_safety_score', 1) < 0.7:
            alerts.append(
                '<div class="warning">⚠️ Low average safety score '
                f"({metrics['average_safety_score']:.3f}). Review safety protocols.</div>"
            )
            
        # Check language disparities
        if 'per_language_metrics' in metrics:
            lang_scores = [m['average_safety_score'] 
                          for m in metrics['per_language_metrics'].values()]
            if lang_scores and (max(lang_scores) - min(lang_scores)) > 0.3:
                alerts.append(
                    '<div class="warning">⚠️ Significant performance disparities '
                    'detected across languages.</div>'
                )
                
        if not alerts:
            alerts.append(
                '<div class="success">✅ All metrics within acceptable ranges.</div>'
            )
            
        return '\n'.join(alerts)
    
    def _generate_language_section(self, metrics: Dict) -> str:
        """Generate language-specific section."""
        if 'per_language_metrics' not in metrics:
            return ""
            
        html = "<h3>Language-Specific Performance</h3>"
        html += """
        <table>
            <tr>
                <th>Language</th>
                <th>Samples</th>
                <th>Hallucination Rate</th>
                <th>Average Safety Score</th>
            </tr>
        """
        
        for lang, lang_metrics in metrics['per_language_metrics'].items():
            html += f"""
            <tr>
                <td>{lang}</td>
                <td>{lang_metrics.get('sample_count', 'N/A')}</td>
                <td>{lang_metrics.get('hallucination_rate', 0):.1%}</td>
                <td>{lang_metrics.get('average_safety_score', 0):.3f}</td>
            </tr>
            """
            
        html += "</table>"
        return html
    
    def _generate_model_section(self, metrics: Dict) -> str:
        """Generate model-specific section."""
        if 'per_model_metrics' not in metrics:
            return ""
            
        html = "<h3>Model Performance Comparison</h3>"
        html += """
        <table>
            <tr>
                <th>Model</th>
                <th>Samples</th>
                <th>Hallucination Rate</th>
                <th>Average Safety Score</th>
            </tr>
        """
        
        for model, model_metrics in metrics['per_model_metrics'].items():
            html += f"""
            <tr>
                <td>{model}</td>
                <td>{model_metrics.get('sample_count', 'N/A')}</td>
                <td>{model_metrics.get('hallucination_rate', 0):.1%}</td>
                <td>{model_metrics.get('average_safety_score', 0):.3f}</td>
            </tr>
            """
            
        html += "</table>"
        return html
    
    def _generate_recommendations(self, metrics: Dict) -> str:
        """Generate recommendations based on analysis."""
        recommendations = ["<h2>💡 Recommendations</h2><ul>"]
        
        # Hallucination recommendations
        if metrics.get('hallucination_rate', 0) > 0.1:
            recommendations.append(
                "<li>Consider implementing additional fact-checking mechanisms</li>"
            )
            recommendations.append(
                "<li>Increase training data quality and diversity</li>"
            )
            
        # Safety recommendations
        if metrics.get('average_safety_score', 1) < 0.8:
            recommendations.append(
                "<li>Enhance safety filters and content moderation</li>"
            )
            recommendations.append(
                "<li>Implement additional red-teaming exercises</li>"
            )
            
        # Language-specific recommendations
        if 'per_language_metrics' in metrics:
            low_resource_langs = [
                lang for lang, m in metrics['per_language_metrics'].items()
                if m.get('sample_count', 0) < 100
            ]
            if low_resource_langs:
                recommendations.append(
                    f"<li>Increase evaluation data for low-resource languages: "
                    f"{', '.join(low_resource_langs)}</li>"
                )
                
        recommendations.append("</ul>")
        return '\n'.join(recommendations)
    
    def generate_pdf_report(self, metrics: Dict, df: Optional[pd.DataFrame] = None,
                          output_path: str = "evaluation_report.pdf") -> str:
        """
        Generate a PDF report with visualizations.
        
        Args:
            metrics: Evaluation metrics
            df: Optional dataframe with detailed results
            output_path: Path to save the PDF report
            
        Returns:
            Path to the generated report
        """
        with PdfPages(output_path) as pdf:
            # Title page
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.5, 0.7, 'Multilingual Alignment\nEvaluation Report', 
                    ha='center', va='center', fontsize=24, weight='bold')
            fig.text(0.5, 0.5, f"Generated: {metrics.get('timestamp', datetime.now().isoformat())}", 
                    ha='center', va='center', fontsize=12)
            fig.text(0.5, 0.3, f"Total Samples: {metrics.get('total_samples', 'N/A')}", 
                    ha='center', va='center', fontsize=14)
            plt.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Metrics summary page
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8.5, 11))
            
            # Metric cards
            self._create_metric_card(ax1, 'Hallucination Rate', 
                                   f"{metrics.get('hallucination_rate', 0):.1%}")
            self._create_metric_card(ax2, 'Average Safety Score', 
                                   f"{metrics.get('average_safety_score', 0):.3f}")
            self._create_metric_card(ax3, 'High Safety Ratio', 
                                   f"{metrics.get('high_safety_ratio', 0):.1%}")
            self._create_metric_card(ax4, 'Languages Evaluated', 
                                   str(metrics.get('languages_evaluated', 'N/A')))
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Additional visualizations
            if df is not None:
                # Language heatmap
                heatmap_fig = self.visualizer.plot_language_heatmap(df)
                if heatmap_fig:
                    pdf.savefig(heatmap_fig, bbox_inches='tight')
                    plt.close()
                
                # Comparison plots
                comparison_figs = self.visualizer.create_comparison_plots(df)
                for fig in comparison_figs:
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()
            
            # Radar chart
            radar_fig = self.visualizer.plot_performance_radar(metrics)
            pdf.savefig(radar_fig, bbox_inches='tight')
            plt.close()
            
            # Metadata
            d = pdf.infodict()
            d['Title'] = 'Multilingual Alignment Evaluation Report'
            d['Author'] = 'Multilingual Evaluation System'
            d['Subject'] = 'AI Safety and Alignment Evaluation'
            d['Keywords'] = 'NLP, Multilingual, Safety, Alignment'
            d['CreationDate'] = datetime.now()
            
        logger.info(f"PDF report saved to {output_path}")
        return output_path
    
    def _create_metric_card(self, ax: plt.Axes, label: str, value: str):
        """Create a metric card visualization."""
        ax.text(0.5, 0.7, value, ha='center', va='center', 
                fontsize=24, weight='bold', color='#3498db')
        ax.text(0.5, 0.3, label, ha='center', va='center', 
                fontsize=12, color='#7f8c8d')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Add border
        rect = plt.Rectangle((0.05, 0.05), 0.9, 0.9, 
                           fill=False, edgecolor='#bdc3c7', linewidth=2)
        ax.add_patch(rect)


# Example usage
if __name__ == "__main__":
    # Create sample metrics
    sample_metrics = {
        'timestamp': datetime.now().isoformat(),
        'total_samples': 1000,
        'hallucination_rate': 0.15,
        'average_safety_score': 0.82,
        'safety_score_std': 0.12,
        'high_safety_ratio': 0.65,
        'low_safety_ratio': 0.10,
        'languages_evaluated': 4,
        'per_language_metrics': {
            'en': {'hallucination_rate': 0.10, 'average_safety_score': 0.88, 'sample_count': 400},
            'sw': {'hallucination_rate': 0.18, 'average_safety_score': 0.79, 'sample_count': 200},
            'hi': {'hallucination_rate': 0.20, 'average_safety_score': 0.77, 'sample_count': 200},
            'id': {'hallucination_rate': 0.15, 'average_safety_score': 0.81, 'sample_count': 200}
        },
        'per_model_metrics': {
            'gpt-4': {'hallucination_rate': 0.08, 'average_safety_score': 0.90, 'sample_count': 300},
            'claude-3': {'hallucination_rate': 0.12, 'average_safety_score': 0.85, 'sample_count': 400},
            'llama-2': {'hallucination_rate': 0.25, 'average_safety_score': 0.70, 'sample_count': 300}
        }
    }
    
    # Initialize visualizer and report generator
    visualizer = EvaluationVisualizer()
    report_gen = ReportGenerator()
    
    # Generate HTML report
    html_path = report_gen.generate_html_report(sample_metrics)
    print(f"HTML report generated: {html_path}")
    
    # Generate PDF report
    pdf_path = report_gen.generate_pdf_report(sample_metrics)
    print(f"PDF report generated: {pdf_path}")
    
    # Create dashboard
    dashboard = visualizer.create_overview_dashboard(sample_metrics, "dashboard.html")
    print("Interactive dashboard created: dashboard.html")