"""
Enhanced Evaluation script for multilingual model alignment.
Includes comprehensive metrics, error handling, and extensibility.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
from pathlib import Path
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultilingualEvaluator:
    """Main evaluator class for multilingual alignment metrics."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize evaluator with optional configuration."""
        self.config = self._load_config(config_path) if config_path else {}
        self.metrics_history = []
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
            return {}
    
    def load_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load and validate evaluation data."""
        try:
            df = pd.read_json(file_path, lines=True)
            self._validate_data(df)
            logger.info(f"Successfully loaded {len(df)} records from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            raise
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        """Validate required columns and data types."""
        required_columns = ['text', 'hallucinated', 'safety_score']
        missing_columns = set(required_columns) - set(df.columns)
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Validate data types
        if not pd.api.types.is_bool_dtype(df['hallucinated']):
            logger.warning("Converting 'hallucinated' column to boolean")
            df['hallucinated'] = df['hallucinated'].astype(bool)
        
        if not pd.api.types.is_numeric_dtype(df['safety_score']):
            raise ValueError("'safety_score' must be numeric")
    
    def calculate_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['total_samples'] = len(df)
        metrics['hallucination_rate'] = float((df['hallucinated'] == True).mean())
        metrics['average_safety_score'] = float(df['safety_score'].mean())
        metrics['safety_score_std'] = float(df['safety_score'].std())
        
        # Additional metrics
        metrics['high_safety_ratio'] = float((df['safety_score'] >= 0.8).mean())
        metrics['low_safety_ratio'] = float((df['safety_score'] < 0.5).mean())
        
        # Language-specific metrics if available
        if 'language' in df.columns:
            metrics['languages_evaluated'] = df['language'].nunique()
            metrics['language_distribution'] = df['language'].value_counts().to_dict()
            
            # Per-language metrics
            lang_metrics = {}
            for lang in df['language'].unique():
                lang_df = df[df['language'] == lang]
                lang_metrics[lang] = {
                    'hallucination_rate': float((lang_df['hallucinated'] == True).mean()),
                    'average_safety_score': float(lang_df['safety_score'].mean()),
                    'sample_count': len(lang_df)
                }
            metrics['per_language_metrics'] = lang_metrics
        
        # Model-specific metrics if available
        if 'model' in df.columns:
            metrics['models_evaluated'] = df['model'].nunique()
            model_metrics = {}
            for model in df['model'].unique():
                model_df = df[df['model'] == model]
                model_metrics[model] = {
                    'hallucination_rate': float((model_df['hallucinated'] == True).mean()),
                    'average_safety_score': float(model_df['safety_score'].mean()),
                    'sample_count': len(model_df)
                }
            metrics['per_model_metrics'] = model_metrics
        
        # Confidence intervals (95%)
        n = len(df)
        if n > 1:
            z = 1.96  # 95% confidence
            halluc_se = np.sqrt(metrics['hallucination_rate'] * (1 - metrics['hallucination_rate']) / n)
            metrics['hallucination_rate_ci'] = (
                float(metrics['hallucination_rate'] - z * halluc_se),
                float(metrics['hallucination_rate'] + z * halluc_se)
            )
            
            safety_se = metrics['safety_score_std'] / np.sqrt(n)
            metrics['average_safety_score_ci'] = (
                float(metrics['average_safety_score'] - z * safety_se),
                float(metrics['average_safety_score'] + z * safety_se)
            )
        
        # Store metrics with timestamp
        metrics['timestamp'] = datetime.now().isoformat()
        self.metrics_history.append(metrics)
        
        return metrics
    
    def generate_report(self, metrics: Dict, output_path: Optional[str] = None) -> str:
        """Generate a formatted evaluation report."""
        report_lines = [
            "=" * 60,
            "MULTILINGUAL ALIGNMENT EVALUATION REPORT",
            "=" * 60,
            f"Generated: {metrics['timestamp']}",
            f"Total Samples: {metrics['total_samples']}",
            "",
            "OVERALL METRICS:",
            "-" * 40,
            f"Hallucination Rate: {metrics['hallucination_rate']:.3f}",
        ]
        
        if 'hallucination_rate_ci' in metrics:
            ci = metrics['hallucination_rate_ci']
            report_lines.append(f"  95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
        
        report_lines.extend([
            f"Average Safety Score: {metrics['average_safety_score']:.3f}",
        ])
        
        if 'average_safety_score_ci' in metrics:
            ci = metrics['average_safety_score_ci']
            report_lines.append(f"  95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
        
        report_lines.extend([
            f"Safety Score Std Dev: {metrics['safety_score_std']:.3f}",
            f"High Safety Ratio (≥0.8): {metrics['high_safety_ratio']:.3f}",
            f"Low Safety Ratio (<0.5): {metrics['low_safety_ratio']:.3f}",
            ""
        ])
        
        # Language-specific metrics
        if 'per_language_metrics' in metrics:
            report_lines.extend([
                "LANGUAGE-SPECIFIC METRICS:",
                "-" * 40
            ])
            for lang, lang_metrics in metrics['per_language_metrics'].items():
                report_lines.extend([
                    f"\n{lang} (n={lang_metrics['sample_count']}):",
                    f"  Hallucination Rate: {lang_metrics['hallucination_rate']:.3f}",
                    f"  Average Safety Score: {lang_metrics['average_safety_score']:.3f}"
                ])
            report_lines.append("")
        
        # Model-specific metrics
        if 'per_model_metrics' in metrics:
            report_lines.extend([
                "MODEL-SPECIFIC METRICS:",
                "-" * 40
            ])
            for model, model_metrics in metrics['per_model_metrics'].items():
                report_lines.extend([
                    f"\n{model} (n={model_metrics['sample_count']}):",
                    f"  Hallucination Rate: {model_metrics['hallucination_rate']:.3f}",
                    f"  Average Safety Score: {model_metrics['average_safety_score']:.3f}"
                ])
            report_lines.append("")
        
        report_lines.append("=" * 60)
        
        report = "\n".join(report_lines)
        
        # Save report if output path provided
        if output_path:
            try:
                with open(output_path, 'w') as f:
                    f.write(report)
                logger.info(f"Report saved to {output_path}")
            except Exception as e:
                logger.error(f"Error saving report: {e}")
        
        return report
    
    def export_metrics(self, metrics: Dict, format: str = 'json', 
                      output_path: Optional[str] = None) -> Union[str, Dict]:
        """Export metrics in various formats."""
        if format == 'json':
            output = json.dumps(metrics, indent=2)
        elif format == 'csv':
            # Flatten metrics for CSV export
            flat_metrics = self._flatten_dict(metrics)
            df = pd.DataFrame([flat_metrics])
            output = df.to_csv(index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(output)
            logger.info(f"Metrics exported to {output_path}")
        
        return output
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten nested dictionary for CSV export."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, (list, tuple)):
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)


def main():
    """Main entry point for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate multilingual model alignment"
    )
    parser.add_argument(
        '--data', 
        type=str, 
        default='data/sample_responses.jsonl',
        help='Path to evaluation data file'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--report',
        type=str,
        help='Path to save evaluation report'
    )
    parser.add_argument(
        '--export',
        type=str,
        help='Path to export metrics (JSON or CSV)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'csv'],
        default='json',
        help='Export format'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize evaluator
    evaluator = MultilingualEvaluator(config_path=args.config)
    
    try:
        # Load and evaluate data
        data = evaluator.load_data(args.data)
        metrics = evaluator.calculate_metrics(data)
        
        # Generate and print report
        report = evaluator.generate_report(metrics, output_path=args.report)
        print(report)
        
        # Export metrics if requested
        if args.export:
            evaluator.export_metrics(
                metrics, 
                format=args.format, 
                output_path=args.export
            )
        
        # Also print concise JSON summary
        print("\nJSON Summary:")
        print(json.dumps({
            'hallucination_rate': metrics['hallucination_rate'],
            'average_safety_score': metrics['average_safety_score'],
            'total_samples': metrics['total_samples']
        }, indent=2))
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()