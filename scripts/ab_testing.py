"""
A/B testing framework for comparing model performance.
Supports statistical significance testing and automated decision making.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import scipy.stats as stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
import json
import uuid
import logging
from enum import Enum
import asyncio
from collections import defaultdict

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Status of an A/B test experiment."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class MetricType(Enum):
    """Types of metrics for A/B testing."""
    BINARY = "binary"  # Success/failure (e.g., hallucination yes/no)
    CONTINUOUS = "continuous"  # Numeric values (e.g., safety score)
    COUNT = "count"  # Count data (e.g., number of errors)
    LATENCY = "latency"  # Response time metrics


@dataclass
class Variant:
    """Represents a variant in an A/B test."""
    name: str
    model_config: Dict
    allocation: float  # Traffic allocation percentage
    is_control: bool = False
    metadata: Optional[Dict] = None


@dataclass
class MetricDefinition:
    """Definition of a metric to track."""
    name: str
    type: MetricType
    success_criteria: Optional[str] = None  # e.g., "increase", "decrease"
    minimum_detectable_effect: float = 0.05  # 5% change
    confidence_level: float = 0.95
    power: float = 0.8


@dataclass
class ExperimentResult:
    """Results from an A/B test experiment."""
    experiment_id: str
    status: ExperimentStatus
    start_time: datetime
    end_time: Optional[datetime]
    variants: Dict[str, Variant]
    metrics: Dict[str, Dict[str, Union[float, Dict]]]  # metric -> variant -> value
    statistical_results: Dict[str, Dict]
    recommendation: Optional[str] = None
    confidence_intervals: Optional[Dict] = None


class ABTestExperiment:
    """Manages a single A/B test experiment."""
    
    def __init__(self, 
                 experiment_id: str,
                 name: str,
                 variants: List[Variant],
                 metrics: List[MetricDefinition],
                 min_sample_size: int = 1000,
                 max_duration_days: int = 30):
        """
        Initialize an A/B test experiment.
        
        Args:
            experiment_id: Unique identifier
            name: Human-readable name
            variants: List of variants to test
            metrics: List of metrics to track
            min_sample_size: Minimum samples per variant
            max_duration_days: Maximum test duration
        """
        self.experiment_id = experiment_id
        self.name = name
        self.variants = {v.name: v for v in variants}
        self.metrics = {m.name: m for m in metrics}
        self.min_sample_size = min_sample_size
        self.max_duration_days = max_duration_days
        
        self.status = ExperimentStatus.DRAFT
        self.start_time = None
        self.data = defaultdict(list)  # variant -> list of observations
        
        # Validate experiment setup
        self._validate_setup()
        
    def _validate_setup(self):
        """Validate experiment configuration."""
        # Check allocations sum to 1
        total_allocation = sum(v.allocation for v in self.variants.values())
        if abs(total_allocation - 1.0) > 0.001:
            raise ValueError(f"Variant allocations must sum to 1.0, got {total_allocation}")
            
        # Ensure exactly one control
        controls = [v for v in self.variants.values() if v.is_control]
        if len(controls) != 1:
            raise ValueError(f"Exactly one control variant required, got {len(controls)}")
            
    def start(self):
        """Start the experiment."""
        if self.status != ExperimentStatus.DRAFT:
            raise RuntimeError(f"Cannot start experiment in {self.status} status")
            
        self.status = ExperimentStatus.RUNNING
        self.start_time = datetime.now()
        logger.info(f"Started experiment {self.experiment_id}")
        
    def stop(self):
        """Stop the experiment."""
        if self.status != ExperimentStatus.RUNNING:
            raise RuntimeError(f"Cannot stop experiment in {self.status} status")
            
        self.status = ExperimentStatus.COMPLETED
        logger.info(f"Stopped experiment {self.experiment_id}")
        
    def pause(self):
        """Pause the experiment."""
        if self.status != ExperimentStatus.RUNNING:
            raise RuntimeError(f"Cannot pause experiment in {self.status} status")
            
        self.status = ExperimentStatus.PAUSED
        logger.info(f"Paused experiment {self.experiment_id}")
        
    def resume(self):
        """Resume a paused experiment."""
        if self.status != ExperimentStatus.PAUSED:
            raise RuntimeError(f"Cannot resume experiment in {self.status} status")
            
        self.status = ExperimentStatus.RUNNING
        logger.info(f"Resumed experiment {self.experiment_id}")
        
    def assign_variant(self, user_id: Optional[str] = None) -> str:
        """
        Assign a user to a variant based on allocation.
        
        Args:
            user_id: Optional user identifier for consistent assignment
            
        Returns:
            Variant name
        """
        if self.status != ExperimentStatus.RUNNING:
            raise RuntimeError(f"Cannot assign variants when experiment is {self.status}")
            
        # Use user_id for consistent assignment if provided
        if user_id:
            # Simple hash-based assignment
            hash_value = int(hash(f"{self.experiment_id}:{user_id}") % 100) / 100
        else:
            hash_value = np.random.random()
            
        # Assign based on allocation
        cumulative = 0
        for variant_name, variant in self.variants.items():
            cumulative += variant.allocation
            if hash_value < cumulative:
                return variant_name
                
        # Fallback (should not happen)
        return list(self.variants.keys())[-1]
        
    def record_observation(self, variant_name: str, metrics: Dict[str, Union[float, bool]]):
        """
        Record an observation for a variant.
        
        Args:
            variant_name: Name of the variant
            metrics: Dictionary of metric values
        """
        if variant_name not in self.variants:
            raise ValueError(f"Unknown variant: {variant_name}")
            
        observation = {
            'timestamp': datetime.now(),
            'variant': variant_name,
            **metrics
        }
        
        self.data[variant_name].append(observation)
        
    def should_stop_early(self) -> Tuple[bool, Optional[str]]:
        """
        Check if experiment should stop early.
        
        Returns:
            Tuple of (should_stop, reason)
        """
        # Check duration
        if self.start_time:
            duration = datetime.now() - self.start_time
            if duration > timedelta(days=self.max_duration_days):
                return True, "Maximum duration reached"
                
        # Check sample size
        min_samples = min(len(self.data[v]) for v in self.variants)
        if min_samples >= self.min_sample_size * 2:
            # Check for overwhelming evidence
            p_values = self._calculate_p_values()
            if any(p < 0.001 for p in p_values.values()):
                return True, "Overwhelming statistical evidence"
                
        return False, None
        
    def _calculate_p_values(self) -> Dict[str, float]:
        """Calculate p-values for all metrics."""
        p_values = {}
        control_name = [v for v in self.variants if self.variants[v].is_control][0]
        
        for metric_name, metric_def in self.metrics.items():
            control_data = [obs[metric_name] for obs in self.data[control_name] 
                          if metric_name in obs]
            
            for variant_name in self.variants:
                if variant_name == control_name:
                    continue
                    
                variant_data = [obs[metric_name] for obs in self.data[variant_name]
                              if metric_name in obs]
                
                if not control_data or not variant_data:
                    continue
                    
                # Calculate p-value based on metric type
                if metric_def.type == MetricType.BINARY:
                    p_value = self._binary_test(control_data, variant_data)
                elif metric_def.type in [MetricType.CONTINUOUS, MetricType.LATENCY]:
                    p_value = self._continuous_test(control_data, variant_data)
                else:
                    p_value = 1.0  # Default to no significance
                    
                p_values[f"{metric_name}:{variant_name}"] = p_value
                
        return p_values
        
    def _binary_test(self, control: List[bool], treatment: List[bool]) -> float:
        """Perform statistical test for binary metrics."""
        # Create contingency table
        control_success = sum(control)
        control_failure = len(control) - control_success
        treatment_success = sum(treatment)
        treatment_failure = len(treatment) - treatment_success
        
        contingency_table = [
            [control_success, control_failure],
            [treatment_success, treatment_failure]
        ]
        
        # Chi-square test
        _, p_value, _, _ = chi2_contingency(contingency_table)
        return p_value
        
    def _continuous_test(self, control: List[float], treatment: List[float]) -> float:
        """Perform statistical test for continuous metrics."""
        # Check normality
        if len(control) > 30 and len(treatment) > 30:
            # Use t-test for large samples
            _, p_value = ttest_ind(control, treatment)
        else:
            # Use Mann-Whitney U test for small samples
            _, p_value = mannwhitneyu(control, treatment, alternative='two-sided')
            
        return p_value
        
    def calculate_results(self) -> ExperimentResult:
        """Calculate experiment results."""
        if self.status not in [ExperimentStatus.COMPLETED, ExperimentStatus.RUNNING]:
            raise RuntimeError(f"Cannot calculate results for {self.status} experiment")
            
        # Calculate metrics for each variant
        metrics_results = {}
        for metric_name, metric_def in self.metrics.items():
            metrics_results[metric_name] = {}
            
            for variant_name in self.variants:
                data = [obs[metric_name] for obs in self.data[variant_name]
                       if metric_name in obs]
                
                if not data:
                    continue
                    
                if metric_def.type == MetricType.BINARY:
                    metrics_results[metric_name][variant_name] = {
                        'value': np.mean(data),
                        'count': len(data),
                        'std_error': np.std(data) / np.sqrt(len(data))
                    }
                else:
                    metrics_results[metric_name][variant_name] = {
                        'mean': np.mean(data),
                        'median': np.median(data),
                        'std': np.std(data),
                        'count': len(data),
                        'percentiles': {
                            'p25': np.percentile(data, 25),
                            'p75': np.percentile(data, 75),
                            'p95': np.percentile(data, 95)
                        }
                    }
                    
        # Calculate statistical significance
        p_values = self._calculate_p_values()
        statistical_results = self._calculate_statistical_significance(p_values)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(metrics_results, statistical_results)
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals()
        
        return ExperimentResult(
            experiment_id=self.experiment_id,
            status=self.status,
            start_time=self.start_time,
            end_time=datetime.now() if self.status == ExperimentStatus.COMPLETED else None,
            variants=self.variants,
            metrics=metrics_results,
            statistical_results=statistical_results,
            recommendation=recommendation,
            confidence_intervals=confidence_intervals
        )
        
    def _calculate_statistical_significance(self, p_values: Dict[str, float]) -> Dict:
        """Calculate statistical significance for all comparisons."""
        results = {}
        
        for key, p_value in p_values.items():
            metric_name, variant_name = key.split(':')
            
            if metric_name not in results:
                results[metric_name] = {}
                
            results[metric_name][variant_name] = {
                'p_value': p_value,
                'is_significant': p_value < (1 - self.metrics[metric_name].confidence_level),
                'confidence_level': self.metrics[metric_name].confidence_level
            }
            
        return results
        
    def _calculate_confidence_intervals(self) -> Dict:
        """Calculate confidence intervals for metric differences."""
        intervals = {}
        control_name = [v for v in self.variants if self.variants[v].is_control][0]
        
        for metric_name in self.metrics:
            intervals[metric_name] = {}
            
            control_data = [obs[metric_name] for obs in self.data[control_name]
                          if metric_name in obs]
            
            if not control_data:
                continue
                
            control_mean = np.mean(control_data)
            control_std = np.std(control_data)
            
            for variant_name in self.variants:
                if variant_name == control_name:
                    continue
                    
                variant_data = [obs[metric_name] for obs in self.data[variant_name]
                              if metric_name in obs]
                
                if not variant_data:
                    continue
                    
                variant_mean = np.mean(variant_data)
                variant_std = np.std(variant_data)
                
                # Calculate pooled standard error
                n1, n2 = len(control_data), len(variant_data)
                pooled_se = np.sqrt((control_std**2 / n1) + (variant_std**2 / n2))
                
                # Calculate confidence interval
                z_score = stats.norm.ppf((1 + self.metrics[metric_name].confidence_level) / 2)
                diff = variant_mean - control_mean
                margin = z_score * pooled_se
                
                intervals[metric_name][variant_name] = {
                    'difference': diff,
                    'lower': diff - margin,
                    'upper': diff + margin,
                    'relative_change': diff / control_mean if control_mean != 0 else np.inf
                }
                
        return intervals
        
    def _generate_recommendation(self, metrics: Dict, stats: Dict) -> str:
        """Generate recommendation based on results."""
        control_name = [v for v in self.variants if self.variants[v].is_control][0]
        
        # Count significant improvements
        improvements = 0
        regressions = 0
        
        for metric_name, metric_stats in stats.items():
            metric_def = self.metrics[metric_name]
            
            for variant_name, variant_stats in metric_stats.items():
                if not variant_stats['is_significant']:
                    continue
                    
                # Check if improvement or regression
                control_value = metrics[metric_name][control_name]['value']
                variant_value = metrics[metric_name][variant_name]['value']
                
                if metric_def.success_criteria == "increase":
                    if variant_value > control_value:
                        improvements += 1
                    else:
                        regressions += 1
                elif metric_def.success_criteria == "decrease":
                    if variant_value < control_value:
                        improvements += 1
                    else:
                        regressions += 1
                        
        # Generate recommendation
        if improvements > 0 and regressions == 0:
            return "Strong evidence to adopt treatment variant"
        elif improvements > regressions:
            return "Moderate evidence to adopt treatment variant"
        elif regressions > improvements:
            return "Evidence suggests keeping control variant"
        else:
            return "Insufficient evidence to make a recommendation"


class ABTestingFramework:
    """Manages multiple A/B test experiments."""
    
    def __init__(self):
        self.experiments: Dict[str, ABTestExperiment] = {}
        self.active_experiments: List[str] = []
        
    def create_experiment(self,
                         name: str,
                         control_model: Dict,
                         treatment_models: List[Dict],
                         metrics: List[MetricDefinition],
                         traffic_split: Optional[List[float]] = None) -> str:
        """
        Create a new A/B test experiment.
        
        Args:
            name: Experiment name
            control_model: Configuration for control model
            treatment_models: List of treatment model configurations
            metrics: Metrics to track
            traffic_split: Traffic allocation (defaults to equal split)
            
        Returns:
            Experiment ID
        """
        experiment_id = str(uuid.uuid4())
        
        # Create variants
        num_variants = len(treatment_models) + 1
        if traffic_split is None:
            traffic_split = [1.0 / num_variants] * num_variants
            
        variants = [
            Variant(
                name="control",
                model_config=control_model,
                allocation=traffic_split[0],
                is_control=True
            )
        ]
        
        for i, treatment_config in enumerate(treatment_models):
            variants.append(
                Variant(
                    name=f"treatment_{i+1}",
                    model_config=treatment_config,
                    allocation=traffic_split[i+1],
                    is_control=False
                )
            )
            
        # Create experiment
        experiment = ABTestExperiment(
            experiment_id=experiment_id,
            name=name,
            variants=variants,
            metrics=metrics
        )
        
        self.experiments[experiment_id] = experiment
        logger.info(f"Created experiment {experiment_id}: {name}")
        
        return experiment_id
        
    def start_experiment(self, experiment_id: str):
        """Start an experiment."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Unknown experiment: {experiment_id}")
            
        experiment = self.experiments[experiment_id]
        experiment.start()
        self.active_experiments.append(experiment_id)
        
    def stop_experiment(self, experiment_id: str) -> ExperimentResult:
        """Stop an experiment and get results."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Unknown experiment: {experiment_id}")
            
        experiment = self.experiments[experiment_id]
        experiment.stop()
        
        if experiment_id in self.active_experiments:
            self.active_experiments.remove(experiment_id)
            
        return experiment.calculate_results()
        
    def get_active_experiments(self) -> List[Dict]:
        """Get list of active experiments."""
        active = []
        for exp_id in self.active_experiments:
            exp = self.experiments[exp_id]
            active.append({
                'id': exp_id,
                'name': exp.name,
                'status': exp.status.value,
                'start_time': exp.start_time.isoformat() if exp.start_time else None,
                'variants': list(exp.variants.keys()),
                'sample_sizes': {v: len(exp.data[v]) for v in exp.variants}
            })
        return active
        
    def route_request(self, experiment_id: str, user_id: Optional[str] = None) -> Dict:
        """
        Route a request to appropriate variant.
        
        Args:
            experiment_id: Experiment to route for
            user_id: Optional user identifier
            
        Returns:
            Routing information including variant and model config
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Unknown experiment: {experiment_id}")
            
        experiment = self.experiments[experiment_id]
        variant_name = experiment.assign_variant(user_id)
        variant = experiment.variants[variant_name]
        
        return {
            'experiment_id': experiment_id,
            'variant': variant_name,
            'model_config': variant.model_config
        }
        
    def record_result(self, experiment_id: str, variant: str, metrics: Dict):
        """Record results from a variant."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Unknown experiment: {experiment_id}")
            
        experiment = self.experiments[experiment_id]
        experiment.record_observation(variant, metrics)
        
        # Check for early stopping
        should_stop, reason = experiment.should_stop_early()
        if should_stop:
            logger.info(f"Early stopping experiment {experiment_id}: {reason}")
            return self.stop_experiment(experiment_id)
            
        return None
        
    def get_experiment_status(self, experiment_id: str) -> Dict:
        """Get current status of an experiment."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Unknown experiment: {experiment_id}")
            
        experiment = self.experiments[experiment_id]
        result = experiment.calculate_results()
        
        return {
            'id': experiment_id,
            'name': experiment.name,
            'status': result.status.value,
            'duration': (datetime.now() - result.start_time).days if result.start_time else 0,
            'metrics': result.metrics,
            'statistical_results': result.statistical_results,
            'recommendation': result.recommendation,
            'confidence_intervals': result.confidence_intervals
        }


# Example usage
if __name__ == "__main__":
    # Create A/B testing framework
    ab_framework = ABTestingFramework()
    
    # Define metrics
    metrics = [
        MetricDefinition(
            name="hallucination_rate",
            type=MetricType.BINARY,
            success_criteria="decrease",
            minimum_detectable_effect=0.05
        ),
        MetricDefinition(
            name="safety_score",
            type=MetricType.CONTINUOUS,
            success_criteria="increase",
            minimum_detectable_effect=0.1
        ),
        MetricDefinition(
            name="response_time",
            type=MetricType.LATENCY,
            success_criteria="decrease",
            minimum_detectable_effect=0.2
        )
    ]
    
    # Create experiment
    experiment_id = ab_framework.create_experiment(
        name="GPT-4 vs Claude-3 Safety Test",
        control_model={'provider': 'openai', 'model': 'gpt-4'},
        treatment_models=[{'provider': 'anthropic', 'model': 'claude-3'}],
        metrics=metrics,
        traffic_split=[0.5, 0.5]
    )
    
    # Start experiment
    ab_framework.start_experiment(experiment_id)
    
    # Simulate some traffic
    np.random.seed(42)
    for i in range(1000):
        # Route request
        routing = ab_framework.route_request(experiment_id, user_id=f"user_{i}")
        
        # Simulate metrics based on variant
        if routing['variant'] == 'control':
            metrics_values = {
                'hallucination_rate': np.random.random() < 0.15,
                'safety_score': np.random.normal(0.8, 0.1),
                'response_time': np.random.exponential(1.5)
            }
        else:
            metrics_values = {
                'hallucination_rate': np.random.random() < 0.10,  # Better
                'safety_score': np.random.normal(0.85, 0.08),     # Better
                'response_time': np.random.exponential(1.8)       # Worse
            }
            
        # Record results
        result = ab_framework.record_result(
            experiment_id, 
            routing['variant'], 
            metrics_values
        )
        
        if result:  # Early stopping triggered
            print("Experiment stopped early")
            break
            
    # Get final results
    if i == 999:  # Didn't stop early
        result = ab_framework.stop_experiment(experiment_id)
        
    # Print results
    status = ab_framework.get_experiment_status(experiment_id)
    print(f"\nExperiment Results:")
    print(f"Recommendation: {status['recommendation']}")
    print(f"\nMetrics:")
    for metric, values in status['metrics'].items():
        print(f"\n{metric}:")
        for variant, stats in values.items():
            print(f"  {variant}: {stats}")
    print(f"\nStatistical Significance:")
    for metric, values in status['statistical_results'].items():
        print(f"\n{metric}:")
        for variant, stats in values.items():
            print(f"  {variant}: p-value={stats['p_value']:.4f}, significant={stats['is_significant']}")