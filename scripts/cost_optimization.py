"""
Cost optimization and analysis module for multilingual evaluation.
Tracks API costs, provides optimization recommendations, and budget controls.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class Provider(Enum):
    """Model providers with different pricing structures."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


@dataclass
class PricingTier:
    """Pricing tier information."""
    name: str
    input_price_per_token: float
    output_price_per_token: float
    min_tokens: int = 0
    max_tokens: Optional[int] = None


@dataclass
class UsageRecord:
    """Record of API usage."""
    timestamp: datetime
    provider: Provider
    model: str
    input_tokens: int
    output_tokens: int
    request_count: int
    latency: float
    success: bool
    language: Optional[str] = None
    experiment_id: Optional[str] = None


@dataclass
class CostAnalysis:
    """Cost analysis results."""
    total_cost: float
    cost_breakdown: Dict[str, float]
    usage_summary: Dict[str, Union[int, float]]
    efficiency_metrics: Dict[str, float]
    recommendations: List[str]
    projected_monthly_cost: float


class PricingCalculator:
    """Calculates costs for different model providers."""
    
    def __init__(self):
        """Initialize with current pricing information."""
        # Pricing as of 2024 (prices in USD per 1M tokens)
        self.pricing = {
            Provider.OPENAI: {
                "gpt-4-turbo": [
                    PricingTier("standard", 10.0, 30.0)
                ],
                "gpt-4": [
                    PricingTier("standard", 30.0, 60.0)
                ],
                "gpt-3.5-turbo": [
                    PricingTier("standard", 0.5, 1.5)
                ]
            },
            Provider.ANTHROPIC: {
                "claude-3-opus": [
                    PricingTier("standard", 15.0, 75.0)
                ],
                "claude-3-sonnet": [
                    PricingTier("standard", 3.0, 15.0)
                ],
                "claude-3-haiku": [
                    PricingTier("standard", 0.25, 1.25)
                ]
            },
            Provider.GOOGLE: {
                "gemini-pro": [
                    PricingTier("standard", 0.5, 1.5)
                ],
                "gemini-ultra": [
                    PricingTier("standard", 2.0, 6.0)
                ]
            },
            Provider.HUGGINGFACE: {
                "default": [
                    PricingTier("standard", 0.1, 0.2)  # Approximate
                ]
            },
            Provider.LOCAL: {
                "default": [
                    PricingTier("compute", 0.0, 0.0)  # No API costs
                ]
            }
        }
        
    def calculate_cost(self, usage: UsageRecord) -> float:
        """Calculate cost for a usage record."""
        if usage.provider == Provider.LOCAL:
            return 0.0
            
        provider_pricing = self.pricing.get(usage.provider, {})
        model_pricing = provider_pricing.get(usage.model)
        
        if not model_pricing:
            # Fallback to default for provider
            model_pricing = provider_pricing.get("default", [
                PricingTier("unknown", 1.0, 2.0)
            ])
            
        # Use first pricing tier (could be extended for volume discounts)
        tier = model_pricing[0]
        
        input_cost = (usage.input_tokens / 1_000_000) * tier.input_price_per_token
        output_cost = (usage.output_tokens / 1_000_000) * tier.output_price_per_token
        
        return input_cost + output_cost
        
    def get_cheapest_option(self, estimated_tokens: Tuple[int, int]) -> Tuple[Provider, str, float]:
        """
        Find the cheapest model option for given token usage.
        
        Args:
            estimated_tokens: (input_tokens, output_tokens)
            
        Returns:
            Tuple of (provider, model, estimated_cost)
        """
        input_tokens, output_tokens = estimated_tokens
        cheapest_cost = float('inf')
        cheapest_option = None
        
        for provider, models in self.pricing.items():
            if provider == Provider.LOCAL:
                continue  # Skip local models for API cost comparison
                
            for model, tiers in models.items():
                tier = tiers[0]  # Use first tier
                
                cost = ((input_tokens / 1_000_000) * tier.input_price_per_token + 
                       (output_tokens / 1_000_000) * tier.output_price_per_token)
                
                if cost < cheapest_cost:
                    cheapest_cost = cost
                    cheapest_option = (provider, model, cost)
                    
        return cheapest_option or (Provider.OPENAI, "gpt-3.5-turbo", 0.0)


class CostTracker:
    """Tracks and analyzes costs over time."""
    
    def __init__(self, budget_limit: Optional[float] = None):
        """
        Initialize cost tracker.
        
        Args:
            budget_limit: Optional monthly budget limit in USD
        """
        self.usage_records: List[UsageRecord] = []
        self.budget_limit = budget_limit
        self.calculator = PricingCalculator()
        self.alerts_sent = set()
        
    def record_usage(self, 
                    provider: Provider,
                    model: str,
                    input_tokens: int,
                    output_tokens: int,
                    latency: float,
                    success: bool = True,
                    language: Optional[str] = None,
                    experiment_id: Optional[str] = None):
        """Record API usage."""
        record = UsageRecord(
            timestamp=datetime.now(),
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            request_count=1,
            latency=latency,
            success=success,
            language=language,
            experiment_id=experiment_id
        )
        
        self.usage_records.append(record)
        self._check_budget_alerts()
        
    def _check_budget_alerts(self):
        """Check if budget alerts should be sent."""
        if not self.budget_limit:
            return
            
        current_cost = self.get_current_month_cost()
        
        # Alert thresholds
        thresholds = [0.5, 0.8, 0.9, 1.0]
        
        for threshold in thresholds:
            if (current_cost >= self.budget_limit * threshold and 
                threshold not in self.alerts_sent):
                
                self.alerts_sent.add(threshold)
                self._send_budget_alert(threshold, current_cost)
                
    def _send_budget_alert(self, threshold: float, current_cost: float):
        """Send budget alert."""
        percentage = threshold * 100
        logger.warning(
            f"Budget Alert: {percentage}% of monthly budget reached. "
            f"Current cost: ${current_cost:.2f}, Budget: ${self.budget_limit:.2f}"
        )
        
    def get_current_month_cost(self) -> float:
        """Get total cost for current month."""
        now = datetime.now()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        month_records = [r for r in self.usage_records if r.timestamp >= month_start]
        return sum(self.calculator.calculate_cost(r) for r in month_records)
        
    def analyze_costs(self, 
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None) -> CostAnalysis:
        """
        Analyze costs over a time period.
        
        Args:
            start_date: Start of analysis period
            end_date: End of analysis period
            
        Returns:
            Cost analysis results
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
            
        # Filter records
        period_records = [
            r for r in self.usage_records 
            if start_date <= r.timestamp <= end_date
        ]
        
        if not period_records:
            return CostAnalysis(
                total_cost=0.0,
                cost_breakdown={},
                usage_summary={},
                efficiency_metrics={},
                recommendations=[],
                projected_monthly_cost=0.0
            )
            
        # Calculate costs
        total_cost = sum(self.calculator.calculate_cost(r) for r in period_records)
        
        # Cost breakdown by provider/model
        cost_breakdown = defaultdict(float)
        for record in period_records:
            key = f"{record.provider.value}:{record.model}"
            cost_breakdown[key] += self.calculator.calculate_cost(record)
            
        # Usage summary
        total_tokens = sum(r.input_tokens + r.output_tokens for r in period_records)
        total_requests = len(period_records)
        avg_latency = np.mean([r.latency for r in period_records])
        success_rate = np.mean([r.success for r in period_records])
        
        usage_summary = {
            'total_tokens': total_tokens,
            'total_requests': total_requests,
            'avg_latency': avg_latency,
            'success_rate': success_rate,
            'cost_per_token': total_cost / max(total_tokens, 1),
            'cost_per_request': total_cost / max(total_requests, 1)
        }
        
        # Efficiency metrics
        efficiency_metrics = self._calculate_efficiency_metrics(period_records)
        
        # Recommendations
        recommendations = self._generate_recommendations(
            period_records, cost_breakdown, efficiency_metrics
        )
        
        # Project monthly cost
        days_in_period = (end_date - start_date).days or 1
        daily_cost = total_cost / days_in_period
        projected_monthly_cost = daily_cost * 30
        
        return CostAnalysis(
            total_cost=total_cost,
            cost_breakdown=dict(cost_breakdown),
            usage_summary=usage_summary,
            efficiency_metrics=efficiency_metrics,
            recommendations=recommendations,
            projected_monthly_cost=projected_monthly_cost
        )
        
    def _calculate_efficiency_metrics(self, records: List[UsageRecord]) -> Dict[str, float]:
        """Calculate efficiency metrics."""
        if not records:
            return {}
            
        # Group by model
        model_groups = defaultdict(list)
        for record in records:
            model_groups[f"{record.provider.value}:{record.model}"].append(record)
            
        metrics = {}
        
        for model_name, model_records in model_groups.items():
            total_cost = sum(self.calculator.calculate_cost(r) for r in model_records)
            total_tokens = sum(r.input_tokens + r.output_tokens for r in model_records)
            avg_latency = np.mean([r.latency for r in model_records])
            success_rate = np.mean([r.success for r in model_records])
            
            # Efficiency score (lower is better)
            efficiency_score = (total_cost * avg_latency) / max(success_rate, 0.1)
            
            metrics[model_name] = {
                'cost_per_token': total_cost / max(total_tokens, 1),
                'avg_latency': avg_latency,
                'success_rate': success_rate,
                'efficiency_score': efficiency_score,
                'usage_count': len(model_records)
            }
            
        return metrics
        
    def _generate_recommendations(self, 
                                records: List[UsageRecord],
                                cost_breakdown: Dict[str, float],
                                efficiency_metrics: Dict[str, float]) -> List[str]:
        """Generate cost optimization recommendations."""
        recommendations = []
        
        if not records:
            return recommendations
            
        # Find most expensive model
        if cost_breakdown:
            most_expensive = max(cost_breakdown.items(), key=lambda x: x[1])
            if most_expensive[1] > sum(cost_breakdown.values()) * 0.5:
                recommendations.append(
                    f"Model {most_expensive[0]} accounts for {most_expensive[1]/sum(cost_breakdown.values()):.1%} "
                    f"of costs. Consider alternatives for non-critical tasks."
                )
                
        # Efficiency recommendations
        if efficiency_metrics:
            efficiency_scores = {k: v['efficiency_score'] for k, v in efficiency_metrics.items()}
            least_efficient = max(efficiency_scores.items(), key=lambda x: x[1])
            
            if len(efficiency_scores) > 1:
                most_efficient = min(efficiency_scores.items(), key=lambda x: x[1])
                
                if least_efficient[1] > most_efficient[1] * 2:
                    recommendations.append(
                        f"Consider replacing {least_efficient[0]} with {most_efficient[0]} "
                        f"for better cost efficiency."
                    )
                    
        # Budget recommendations
        if self.budget_limit:
            current_cost = self.get_current_month_cost()
            if current_cost > self.budget_limit * 0.8:
                recommendations.append(
                    "Approaching budget limit. Consider using smaller models or reducing evaluation frequency."
                )
                
        # Volume discount opportunities
        total_monthly_tokens = sum(
            r.input_tokens + r.output_tokens for r in records
            if r.timestamp >= datetime.now().replace(day=1)
        )
        
        if total_monthly_tokens > 10_000_000:  # 10M tokens
            recommendations.append(
                "High token usage detected. Contact providers about volume discounts."
            )
            
        # Language-specific recommendations
        if any(r.language for r in records):
            lang_costs = defaultdict(float)
            for record in records:
                if record.language:
                    lang_costs[record.language] += self.calculator.calculate_cost(record)
                    
            if lang_costs:
                most_expensive_lang = max(lang_costs.items(), key=lambda x: x[1])
                if most_expensive_lang[1] > sum(lang_costs.values()) * 0.6:
                    recommendations.append(
                        f"Language {most_expensive_lang[0]} has highest costs. "
                        f"Consider optimizing prompts or using smaller models for this language."
                    )
                    
        return recommendations
        
    def optimize_model_selection(self, 
                                task_requirements: Dict[str, float],
                                estimated_tokens: Tuple[int, int]) -> Dict[str, Union[str, float]]:
        """
        Recommend optimal model based on requirements and cost.
        
        Args:
            task_requirements: Dict with 'accuracy', 'speed', 'cost_weight' (0-1)
            estimated_tokens: (input_tokens, output_tokens)
            
        Returns:
            Optimization recommendation
        """
        input_tokens, output_tokens = estimated_tokens
        
        # Score different models
        model_scores = {}
        
        for provider, models in self.calculator.pricing.items():
            if provider == Provider.LOCAL:
                continue
                
            for model, tiers in models.items():
                tier = tiers[0]
                
                # Calculate cost
                cost = ((input_tokens / 1_000_000) * tier.input_price_per_token + 
                       (output_tokens / 1_000_000) * tier.output_price_per_token)
                
                # Estimate performance (simplified)
                performance_scores = {
                    'gpt-4': 0.95,
                    'claude-3-opus': 0.93,
                    'gpt-4-turbo': 0.92,
                    'claude-3-sonnet': 0.88,
                    'gemini-ultra': 0.85,
                    'gpt-3.5-turbo': 0.80,
                    'claude-3-haiku': 0.75,
                    'gemini-pro': 0.70
                }
                
                performance = performance_scores.get(model, 0.6)
                
                # Speed estimate (inverse of typical latency)
                speed_scores = {
                    'gpt-3.5-turbo': 0.95,
                    'claude-3-haiku': 0.90,
                    'gemini-pro': 0.85,
                    'claude-3-sonnet': 0.80,
                    'gpt-4-turbo': 0.75,
                    'gpt-4': 0.60,
                    'claude-3-opus': 0.55,
                    'gemini-ultra': 0.50
                }
                
                speed = speed_scores.get(model, 0.5)
                
                # Normalize cost (lower is better)
                max_cost = 100.0  # Reasonable maximum
                cost_score = 1 - min(cost / max_cost, 1.0)
                
                # Weighted score
                accuracy_weight = task_requirements.get('accuracy', 0.4)
                speed_weight = task_requirements.get('speed', 0.3)
                cost_weight = task_requirements.get('cost_weight', 0.3)
                
                total_score = (
                    performance * accuracy_weight +
                    speed * speed_weight +
                    cost_score * cost_weight
                )
                
                model_scores[f"{provider.value}:{model}"] = {
                    'score': total_score,
                    'cost': cost,
                    'performance': performance,
                    'speed': speed,
                    'provider': provider.value,
                    'model': model
                }
                
        # Find best option
        best_option = max(model_scores.items(), key=lambda x: x[1]['score'])
        
        return {
            'recommended_model': best_option[0],
            'estimated_cost': best_option[1]['cost'],
            'score_breakdown': best_option[1],
            'alternatives': sorted(
                model_scores.items(),
                key=lambda x: x[1]['score'],
                reverse=True
            )[:3]  # Top 3 alternatives
        }
        
    def generate_cost_report(self, output_path: Optional[str] = None) -> str:
        """Generate a comprehensive cost report."""
        analysis = self.analyze_costs()
        
        report_lines = [
            "# Cost Analysis Report",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## Summary",
            f"- Total Cost (30 days): ${analysis.total_cost:.2f}",
            f"- Projected Monthly Cost: ${analysis.projected_monthly_cost:.2f}",
            f"- Total Requests: {analysis.usage_summary.get('total_requests', 0):,}",
            f"- Total Tokens: {analysis.usage_summary.get('total_tokens', 0):,}",
            f"- Average Cost per Request: ${analysis.usage_summary.get('cost_per_request', 0):.4f}",
            f"- Average Cost per Token: ${analysis.usage_summary.get('cost_per_token', 0):.6f}",
            "",
            "## Cost Breakdown by Model",
        ]
        
        for model, cost in sorted(analysis.cost_breakdown.items(), 
                                key=lambda x: x[1], reverse=True):
            percentage = (cost / analysis.total_cost) * 100 if analysis.total_cost > 0 else 0
            report_lines.append(f"- {model}: ${cost:.2f} ({percentage:.1f}%)")
            
        report_lines.extend([
            "",
            "## Efficiency Metrics",
        ])
        
        for model, metrics in analysis.efficiency_metrics.items():
            report_lines.extend([
                f"### {model}",
                f"- Cost per Token: ${metrics['cost_per_token']:.6f}",
                f"- Average Latency: {metrics['avg_latency']:.2f}s",
                f"- Success Rate: {metrics['success_rate']:.1%}",
                f"- Usage Count: {metrics['usage_count']}",
                ""
            ])
            
        if analysis.recommendations:
            report_lines.extend([
                "## Recommendations",
                ""
            ])
            for rec in analysis.recommendations:
                report_lines.append(f"- {rec}")
                
        if self.budget_limit:
            current_cost = self.get_current_month_cost()
            budget_usage = (current_cost / self.budget_limit) * 100
            report_lines.extend([
                "",
                "## Budget Status",
                f"- Monthly Budget: ${self.budget_limit:.2f}",
                f"- Current Usage: ${current_cost:.2f} ({budget_usage:.1f}%)",
                f"- Remaining Budget: ${max(self.budget_limit - current_cost, 0):.2f}"
            ])
            
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
                
        return report
        
    def export_usage_data(self, output_path: str, format: str = 'csv'):
        """Export usage data for external analysis."""
        if not self.usage_records:
            return
            
        # Convert to DataFrame
        data = []
        for record in self.usage_records:
            data.append({
                'timestamp': record.timestamp,
                'provider': record.provider.value,
                'model': record.model,
                'input_tokens': record.input_tokens,
                'output_tokens': record.output_tokens,
                'total_tokens': record.input_tokens + record.output_tokens,
                'cost': self.calculator.calculate_cost(record),
                'latency': record.latency,
                'success': record.success,
                'language': record.language,
                'experiment_id': record.experiment_id
            })
            
        df = pd.DataFrame(data)
        
        if format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'json':
            df.to_json(output_path, orient='records', date_format='iso')
        elif format == 'excel':
            df.to_excel(output_path, index=False)
            
        logger.info(f"Exported {len(data)} usage records to {output_path}")


# Example usage
if __name__ == "__main__":
    # Initialize cost tracker with budget
    tracker = CostTracker(budget_limit=1000.0)  # $1000/month budget
    
    # Simulate some usage
    np.random.seed(42)
    for i in range(100):
        tracker.record_usage(
            provider=Provider.OPENAI,
            model="gpt-4",
            input_tokens=np.random.randint(100, 1000),
            output_tokens=np.random.randint(50, 500),
            latency=np.random.exponential(2.0),
            success=np.random.random() > 0.05,
            language=np.random.choice(['en', 'es', 'zh'])
        )
        
    # Analyze costs
    analysis = tracker.analyze_costs()
    print(f"Total cost: ${analysis.total_cost:.2f}")
    print(f"Projected monthly: ${analysis.projected_monthly_cost:.2f}")
    print("\nRecommendations:")
    for rec in analysis.recommendations:
        print(f"- {rec}")
        
    # Optimize model selection
    optimization = tracker.optimize_model_selection(
        task_requirements={'accuracy': 0.8, 'speed': 0.1, 'cost_weight': 0.1},
        estimated_tokens=(500, 200)
    )
    print(f"\nRecommended model: {optimization['recommended_model']}")
    print(f"Estimated cost: ${optimization['estimated_cost']:.4f}")
    
    # Generate report
    report = tracker.generate_cost_report()
    print("\n" + "="*50)
    print(report)