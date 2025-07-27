"""
Real-time monitoring system for multilingual alignment evaluation.
Provides continuous monitoring, alerting, and performance tracking.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
import smtplib
from email.mime.text import MIMEText
import aiohttp
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics to monitor."""
    HALLUCINATION_RATE = "hallucination_rate"
    SAFETY_SCORE = "safety_score"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    BIAS_SCORE = "bias_score"


@dataclass
class Alert:
    """Alert data structure."""
    timestamp: datetime
    severity: AlertSeverity
    metric: MetricType
    value: float
    threshold: float
    message: str
    language: Optional[str] = None
    model: Optional[str] = None


@dataclass
class MonitorConfig:
    """Monitoring configuration."""
    metrics: Dict[MetricType, Dict[str, float]]  # metric -> {warning, critical} thresholds
    check_interval: int = 300  # seconds
    window_size: int = 3600  # seconds for rolling window
    alert_cooldown: int = 1800  # seconds between similar alerts
    notification_channels: List[str] = None  # email, slack, webhook


class MetricMonitor:
    """Monitors individual metrics."""
    
    def __init__(self, metric_type: MetricType, config: MonitorConfig):
        self.metric_type = metric_type
        self.config = config
        self.history: List[tuple] = []  # (timestamp, value)
        self.last_alert_time: Optional[datetime] = None
        
    def add_measurement(self, value: float, timestamp: Optional[datetime] = None):
        """Add a new measurement."""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.history.append((timestamp, value))
        self._cleanup_old_data()
        
    def _cleanup_old_data(self):
        """Remove data outside the window."""
        cutoff = datetime.now() - timedelta(seconds=self.config.window_size)
        self.history = [(t, v) for t, v in self.history if t > cutoff]
        
    def check_thresholds(self) -> Optional[Alert]:
        """Check if current metrics violate thresholds."""
        if not self.history:
            return None
            
        # Calculate current metric value
        current_value = self._calculate_current_value()
        
        # Get thresholds
        thresholds = self.config.metrics.get(self.metric_type, {})
        
        # Check for violations
        alert = None
        if current_value > thresholds.get('critical', float('inf')):
            alert = self._create_alert(
                AlertSeverity.CRITICAL,
                current_value,
                thresholds['critical']
            )
        elif current_value > thresholds.get('warning', float('inf')):
            alert = self._create_alert(
                AlertSeverity.WARNING,
                current_value,
                thresholds['warning']
            )
            
        # Check cooldown
        if alert and self._should_send_alert():
            self.last_alert_time = datetime.now()
            return alert
            
        return None
        
    def _calculate_current_value(self) -> float:
        """Calculate current metric value from history."""
        if not self.history:
            return 0.0
            
        values = [v for _, v in self.history]
        
        # Different calculation methods based on metric type
        if self.metric_type in [MetricType.HALLUCINATION_RATE, MetricType.ERROR_RATE]:
            return np.mean(values)
        elif self.metric_type == MetricType.SAFETY_SCORE:
            return np.mean(values)
        elif self.metric_type == MetricType.RESPONSE_TIME:
            return np.percentile(values, 95)  # 95th percentile
        else:
            return np.mean(values)
            
    def _create_alert(self, severity: AlertSeverity, value: float, 
                     threshold: float) -> Alert:
        """Create an alert object."""
        return Alert(
            timestamp=datetime.now(),
            severity=severity,
            metric=self.metric_type,
            value=value,
            threshold=threshold,
            message=f"{self.metric_type.value} ({value:.3f}) exceeds {severity.value} threshold ({threshold:.3f})"
        )
        
    def _should_send_alert(self) -> bool:
        """Check if enough time has passed since last alert."""
        if self.last_alert_time is None:
            return True
            
        time_since_last = (datetime.now() - self.last_alert_time).seconds
        return time_since_last > self.config.alert_cooldown
        
    def get_statistics(self) -> Dict:
        """Get current statistics."""
        if not self.history:
            return {}
            
        values = [v for _, v in self.history]
        return {
            'current': values[-1] if values else 0,
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'count': len(values)
        }


class AlertManager:
    """Manages alert notifications."""
    
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.alert_history: List[Alert] = []
        
    async def send_alert(self, alert: Alert):
        """Send alert through configured channels."""
        self.alert_history.append(alert)
        
        for channel in self.config.notification_channels or []:
            if channel == 'email':
                await self._send_email_alert(alert)
            elif channel == 'slack':
                await self._send_slack_alert(alert)
            elif channel.startswith('webhook:'):
                await self._send_webhook_alert(alert, channel[8:])
            elif channel == 'console':
                self._send_console_alert(alert)
                
    def _send_console_alert(self, alert: Alert):
        """Print alert to console."""
        color_map = {
            AlertSeverity.INFO: '\033[94m',
            AlertSeverity.WARNING: '\033[93m',
            AlertSeverity.ERROR: '\033[91m',
            AlertSeverity.CRITICAL: '\033[95m'
        }
        
        color = color_map.get(alert.severity, '')
        reset = '\033[0m'
        
        print(f"{color}[{alert.severity.value.upper()}]{reset} {alert.timestamp}: {alert.message}")
        
    async def _send_email_alert(self, alert: Alert):
        """Send email alert."""
        # Implementation depends on email configuration
        logger.info(f"Email alert: {alert.message}")
        
    async def _send_slack_alert(self, alert: Alert):
        """Send Slack alert."""
        # Implementation depends on Slack webhook
        logger.info(f"Slack alert: {alert.message}")
        
    async def _send_webhook_alert(self, alert: Alert, webhook_url: str):
        """Send alert to webhook."""
        async with aiohttp.ClientSession() as session:
            payload = {
                'timestamp': alert.timestamp.isoformat(),
                'severity': alert.severity.value,
                'metric': alert.metric.value,
                'value': alert.value,
                'threshold': alert.threshold,
                'message': alert.message
            }
            
            try:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status != 200:
                        logger.error(f"Webhook alert failed: {response.status}")
            except Exception as e:
                logger.error(f"Webhook alert error: {e}")


class RealtimeMonitor:
    """Main monitoring system."""
    
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.monitors: Dict[MetricType, MetricMonitor] = {}
        self.alert_manager = AlertManager(config)
        self.running = False
        
        # Initialize monitors
        for metric_type in config.metrics:
            self.monitors[metric_type] = MetricMonitor(metric_type, config)
            
    async def start(self):
        """Start monitoring."""
        self.running = True
        logger.info("Monitoring system started")
        
        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())
        
    async def stop(self):
        """Stop monitoring."""
        self.running = False
        logger.info("Monitoring system stopped")
        
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Check all monitors
                for monitor in self.monitors.values():
                    alert = monitor.check_thresholds()
                    if alert:
                        await self.alert_manager.send_alert(alert)
                        
                # Wait for next check
                await asyncio.sleep(self.config.check_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(10)  # Brief pause on error
                
    def record_metric(self, metric_type: MetricType, value: float,
                     timestamp: Optional[datetime] = None):
        """Record a metric value."""
        if metric_type in self.monitors:
            self.monitors[metric_type].add_measurement(value, timestamp)
            
    def record_evaluation_result(self, result: Dict):
        """Record results from an evaluation."""
        # Extract relevant metrics
        if 'hallucination_rate' in result:
            self.record_metric(
                MetricType.HALLUCINATION_RATE,
                result['hallucination_rate']
            )
            
        if 'average_safety_score' in result:
            self.record_metric(
                MetricType.SAFETY_SCORE,
                result['average_safety_score']
            )
            
        if 'response_time' in result:
            self.record_metric(
                MetricType.RESPONSE_TIME,
                result['response_time']
            )
            
    def get_dashboard_data(self) -> Dict:
        """Get data for dashboard display."""
        return {
            'monitors': {
                metric.value: monitor.get_statistics()
                for metric, monitor in self.monitors.items()
            },
            'recent_alerts': [
                {
                    'timestamp': alert.timestamp.isoformat(),
                    'severity': alert.severity.value,
                    'message': alert.message
                }
                for alert in self.alert_manager.alert_history[-10:]
            ],
            'status': 'running' if self.running else 'stopped'
        }
        
    async def run_health_check(self) -> Dict:
        """Run system health check."""
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'monitors_active': len(self.monitors),
            'alerts_sent': len(self.alert_manager.alert_history),
            'status': 'healthy'
        }
        
        # Check if any monitor has critical issues
        for monitor in self.monitors.values():
            stats = monitor.get_statistics()
            if stats.get('current', 0) > self.config.metrics.get(
                monitor.metric_type, {}
            ).get('critical', float('inf')):
                health_status['status'] = 'critical'
                break
                
        return health_status


# Example usage
if __name__ == "__main__":
    # Configure monitoring
    config = MonitorConfig(
        metrics={
            MetricType.HALLUCINATION_RATE: {'warning': 0.15, 'critical': 0.25},
            MetricType.SAFETY_SCORE: {'warning': 0.7, 'critical': 0.5},
            MetricType.RESPONSE_TIME: {'warning': 2.0, 'critical': 5.0},
            MetricType.ERROR_RATE: {'warning': 0.05, 'critical': 0.1}
        },
        check_interval=60,  # Check every minute
        notification_channels=['console', 'webhook:http://localhost:8001/alerts']
    )
    
    # Create monitor
    monitor = RealtimeMonitor(config)
    
    # Simulate some data
    async def simulate_data():
        await monitor.start()
        
        for i in range(100):
            # Simulate metrics
            monitor.record_metric(
                MetricType.HALLUCINATION_RATE,
                0.1 + np.random.random() * 0.2
            )
            monitor.record_metric(
                MetricType.SAFETY_SCORE,
                0.6 + np.random.random() * 0.3
            )
            monitor.record_metric(
                MetricType.RESPONSE_TIME,
                1.0 + np.random.exponential(1.0)
            )
            
            await asyncio.sleep(5)
            
        await monitor.stop()
        
    # Run simulation
    asyncio.run(simulate_data())