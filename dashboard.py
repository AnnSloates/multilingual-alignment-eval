"""
Interactive web dashboard for multilingual alignment evaluation.
Provides real-time monitoring, analysis, and configuration interface.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sys
import os
from typing import Dict, List, Optional

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from evaluate import MultilingualEvaluator
from data_processing import DataValidator, DataPreprocessor
from prompt_manager import MultilingualPromptManager
from monitoring import RealtimeMonitor, MonitorConfig, MetricType
from bias_detection import BiasDetector, FairnessReportGenerator
from ab_testing import ABTestingFramework, MetricDefinition, MetricType as ABMetricType
from cost_optimization import CostTracker, Provider
from visualization import EvaluationVisualizer, ReportGenerator

# Configure Streamlit page
st.set_page_config(
    page_title="Multilingual Alignment Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-high {
        background-color: #ffe6e6;
        border-left: 4px solid #ff4444;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .alert-medium {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .success {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'evaluator' not in st.session_state:
    st.session_state.evaluator = MultilingualEvaluator()
if 'monitor' not in st.session_state:
    st.session_state.monitor = None
if 'ab_framework' not in st.session_state:
    st.session_state.ab_framework = ABTestingFramework()
if 'cost_tracker' not in st.session_state:
    st.session_state.cost_tracker = CostTracker()

def main():
    """Main dashboard function."""
    st.title("🌍 Multilingual Alignment Evaluation Dashboard")
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Navigate to:",
        [
            "📊 Overview",
            "🔍 Evaluation",
            "📈 Monitoring",
            "⚖️ Bias Analysis",
            "🧪 A/B Testing",
            "💰 Cost Analysis",
            "📝 Reports",
            "⚙️ Settings"
        ]
    )
    
    # Route to appropriate page
    if page == "📊 Overview":
        show_overview()
    elif page == "🔍 Evaluation":
        show_evaluation()
    elif page == "📈 Monitoring":
        show_monitoring()
    elif page == "⚖️ Bias Analysis":
        show_bias_analysis()
    elif page == "🧪 A/B Testing":
        show_ab_testing()
    elif page == "💰 Cost Analysis":
        show_cost_analysis()
    elif page == "📝 Reports":
        show_reports()
    elif page == "⚙️ Settings":
        show_settings()

def show_overview():
    """Show overview dashboard."""
    st.header("Dashboard Overview")
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Accuracy Rate</h3>
            <h2>87.3%</h2>
            <p>+2.1% from last week</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🛡️ Safety Score</h3>
            <h2>0.82</h2>
            <p>-0.03 from last week</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🌐 Languages</h3>
            <h2>8</h2>
            <p>Active evaluations</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>💸 Monthly Cost</h3>
            <h2>$847</h2>
            <p>85% of budget</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Recent alerts
    st.subheader("🚨 Recent Alerts")
    
    # Sample alerts
    alerts = [
        {"level": "high", "message": "Hallucination rate increased to 18% in Hindi", "time": "2 hours ago"},
        {"level": "medium", "message": "API costs approaching budget limit", "time": "5 hours ago"},
        {"level": "low", "message": "New A/B test completed", "time": "1 day ago"}
    ]
    
    for alert in alerts:
        if alert["level"] == "high":
            st.markdown(f"""
            <div class="alert-high">
                <strong>⚠️ High Priority:</strong> {alert["message"]} <em>({alert["time"]})</em>
            </div>
            """, unsafe_allow_html=True)
        elif alert["level"] == "medium":
            st.markdown(f"""
            <div class="alert-medium">
                <strong>⚡ Medium Priority:</strong> {alert["message"]} <em>({alert["time"]})</em>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Performance trends
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Performance Trends")
        
        # Generate sample data
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        safety_scores = 0.8 + 0.1 * np.random.randn(len(dates)).cumsum() * 0.01
        hallucination_rates = 0.15 + 0.05 * np.random.randn(len(dates)).cumsum() * 0.01
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, 
            y=safety_scores,
            mode='lines+markers',
            name='Safety Score',
            line=dict(color='green')
        ))
        fig.add_trace(go.Scatter(
            x=dates, 
            y=hallucination_rates,
            mode='lines+markers',
            name='Hallucination Rate',
            yaxis='y2',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis=dict(title="Safety Score", side="left"),
            yaxis2=dict(title="Hallucination Rate", side="right", overlaying="y"),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🌍 Language Performance")
        
        # Sample language data
        languages = ['English', 'Spanish', 'Chinese', 'Hindi', 'Swahili', 'Arabic']
        scores = [0.89, 0.86, 0.82, 0.78, 0.74, 0.80]
        
        fig = px.bar(
            x=languages, 
            y=scores,
            color=scores,
            color_continuous_scale='RdYlGn',
            title="Average Safety Score by Language"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def show_evaluation():
    """Show evaluation interface."""
    st.header("🔍 Model Evaluation")
    
    tab1, tab2, tab3 = st.tabs(["📄 Data Upload", "⚙️ Configuration", "📊 Results"])
    
    with tab1:
        st.subheader("Upload Evaluation Data")
        
        uploaded_file = st.file_uploader(
            "Choose a JSONL file",
            type=['jsonl', 'json'],
            help="Upload evaluation data in JSONL format"
        )
        
        if uploaded_file is not None:
            try:
                # Load data
                content = uploaded_file.read().decode('utf-8')
                lines = content.strip().split('\n')
                data = [json.loads(line) for line in lines if line.strip()]
                df = pd.DataFrame(data)
                
                st.success(f"✅ Loaded {len(df)} records")
                st.dataframe(df.head())
                
                # Store in session state
                st.session_state.evaluation_data = df
                
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")
    
    with tab2:
        st.subheader("Evaluation Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Metrics")
            metrics = st.multiselect(
                "Select metrics to calculate:",
                ["Hallucination Rate", "Safety Score", "Bias Detection", "Cultural Sensitivity"],
                default=["Hallucination Rate", "Safety Score"]
            )
            
            confidence_level = st.slider(
                "Confidence Level",
                min_value=0.90,
                max_value=0.99,
                value=0.95,
                step=0.01
            )
        
        with col2:
            st.subheader("Filters")
            
            if 'evaluation_data' in st.session_state:
                df = st.session_state.evaluation_data
                
                if 'language' in df.columns:
                    languages = st.multiselect(
                        "Filter by languages:",
                        options=df['language'].unique(),
                        default=df['language'].unique()
                    )
                
                if 'model' in df.columns:
                    models = st.multiselect(
                        "Filter by models:",
                        options=df['model'].unique(),
                        default=df['model'].unique()
                    )
    
    with tab3:
        st.subheader("Evaluation Results")
        
        if st.button("🚀 Run Evaluation"):
            if 'evaluation_data' in st.session_state:
                with st.spinner("Running evaluation..."):
                    # Run evaluation
                    evaluator = st.session_state.evaluator
                    df = st.session_state.evaluation_data
                    
                    # Apply filters if they exist
                    filtered_df = df.copy()
                    
                    try:
                        metrics_result = evaluator.calculate_metrics(filtered_df)
                        
                        # Display results
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Hallucination Rate",
                                f"{metrics_result.get('hallucination_rate', 0):.1%}",
                                delta="-2.3%" if metrics_result.get('hallucination_rate', 0) < 0.15 else "+1.2%"
                            )
                        
                        with col2:
                            st.metric(
                                "Average Safety Score",
                                f"{metrics_result.get('average_safety_score', 0):.3f}",
                                delta="+0.025" if metrics_result.get('average_safety_score', 0) > 0.8 else "-0.015"
                            )
                        
                        with col3:
                            st.metric(
                                "Total Samples",
                                f"{metrics_result.get('total_samples', 0):,}"
                            )
                        
                        # Detailed results
                        st.subheader("Detailed Results")
                        st.json(metrics_result)
                        
                    except Exception as e:
                        st.error(f"❌ Evaluation error: {e}")
            else:
                st.warning("⚠️ Please upload data first")

def show_monitoring():
    """Show real-time monitoring interface."""
    st.header("📈 Real-time Monitoring")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Live Metrics")
        
        # Monitoring controls
        if st.session_state.monitor is None:
            if st.button("▶️ Start Monitoring"):
                # Configure monitoring
                config = MonitorConfig(
                    metrics={
                        MetricType.HALLUCINATION_RATE: {'warning': 0.15, 'critical': 0.25},
                        MetricType.SAFETY_SCORE: {'warning': 0.7, 'critical': 0.5},
                        MetricType.RESPONSE_TIME: {'warning': 2.0, 'critical': 5.0}
                    },
                    check_interval=60,
                    notification_channels=['console']
                )
                st.session_state.monitor = RealtimeMonitor(config)
                st.success("✅ Monitoring started")
                st.experimental_rerun()
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("⏸️ Pause Monitoring"):
                    st.session_state.monitor = None
                    st.info("ℹ️ Monitoring paused")
                    st.experimental_rerun()
            with col_b:
                if st.button("🔄 Refresh Data"):
                    st.experimental_rerun()
        
        # Display metrics if monitoring is active
        if st.session_state.monitor:
            dashboard_data = st.session_state.monitor.get_dashboard_data()
            
            # Metrics grid
            metrics_data = dashboard_data.get('monitors', {})
            
            if metrics_data:
                for metric_name, stats in metrics_data.items():
                    if stats:
                        col_metric = st.columns(4)
                        with col_metric[0]:
                            st.metric(
                                metric_name.replace('_', ' ').title(),
                                f"{stats.get('current', 0):.3f}"
                            )
                        with col_metric[1]:
                            st.metric("Mean", f"{stats.get('mean', 0):.3f}")
                        with col_metric[2]:
                            st.metric("Max", f"{stats.get('max', 0):.3f}")
                        with col_metric[3]:
                            st.metric("Count", stats.get('count', 0))
            
            # Recent alerts
            recent_alerts = dashboard_data.get('recent_alerts', [])
            if recent_alerts:
                st.subheader("🚨 Recent Alerts")
                for alert in recent_alerts[-5:]:  # Show last 5
                    st.warning(f"**{alert['timestamp']}**: {alert['message']}")
    
    with col2:
        st.subheader("⚙️ Monitoring Settings")
        
        st.subheader("Thresholds")
        
        halluc_warning = st.number_input(
            "Hallucination Warning",
            min_value=0.0,
            max_value=1.0,
            value=0.15,
            step=0.01
        )
        
        halluc_critical = st.number_input(
            "Hallucination Critical",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.01
        )
        
        safety_warning = st.number_input(
            "Safety Score Warning",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.01
        )
        
        check_interval = st.number_input(
            "Check Interval (seconds)",
            min_value=10,
            max_value=3600,
            value=60
        )
        
        if st.button("💾 Update Settings"):
            st.success("✅ Settings updated")

def show_bias_analysis():
    """Show bias analysis interface."""
    st.header("⚖️ Bias & Fairness Analysis")
    
    tab1, tab2, tab3 = st.tabs(["🔍 Detection", "📊 Analysis", "📋 Report"])
    
    with tab1:
        st.subheader("Bias Detection")
        
        # Text input for analysis
        text_input = st.text_area(
            "Enter text to analyze for bias:",
            height=150,
            placeholder="Enter model output text here..."
        )
        
        language = st.selectbox(
            "Select language:",
            ["en", "es", "zh", "hi", "sw", "ar", "fr"]
        )
        
        if st.button("🔍 Analyze Bias") and text_input:
            detector = BiasDetector()
            bias_scores = detector.analyze_text(text_input, language)
            
            if bias_scores:
                st.subheader("📊 Bias Analysis Results")
                
                for score in bias_scores:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            f"{score.bias_type.value.title()} Bias",
                            f"{score.score:.3f}"
                        )
                    
                    with col2:
                        st.metric("Confidence", f"{score.confidence:.2f}")
                    
                    with col3:
                        if score.score > 0.7:
                            st.error("🔴 High Risk")
                        elif score.score > 0.4:
                            st.warning("🟡 Moderate Risk")
                        else:
                            st.success("🟢 Low Risk")
                    
                    if score.examples:
                        with st.expander(f"View {score.bias_type.value} examples"):
                            st.json(score.examples)
            else:
                st.info("ℹ️ No significant bias detected")
    
    with tab2:
        st.subheader("Fairness Analysis")
        
        if 'evaluation_data' in st.session_state:
            df = st.session_state.evaluation_data
            
            if st.button("🔍 Run Fairness Analysis"):
                report_gen = FairnessReportGenerator()
                
                with st.spinner("Analyzing fairness..."):
                    # Analyze bias by language
                    detector = BiasDetector()
                    
                    bias_by_lang = {}
                    if 'language' in df.columns:
                        for lang in df['language'].unique():
                            lang_df = df[df['language'] == lang]
                            lang_scores = []
                            
                            for text in lang_df['text'].head(10):  # Sample for demo
                                scores = detector.analyze_text(str(text), lang)
                                if scores:
                                    lang_scores.extend([s.score for s in scores])
                            
                            if lang_scores:
                                bias_by_lang[lang] = {
                                    'mean_bias': np.mean(lang_scores),
                                    'max_bias': np.max(lang_scores),
                                    'count': len(lang_scores)
                                }
                    
                    # Display results
                    if bias_by_lang:
                        st.subheader("📊 Bias by Language")
                        
                        lang_names = list(bias_by_lang.keys())
                        mean_scores = [bias_by_lang[lang]['mean_bias'] for lang in lang_names]
                        
                        fig = px.bar(
                            x=lang_names,
                            y=mean_scores,
                            color=mean_scores,
                            color_continuous_scale='RdYlGn_r',
                            title="Average Bias Score by Language"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed breakdown
                        for lang, stats in bias_by_lang.items():
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(f"{lang} - Mean Bias", f"{stats['mean_bias']:.3f}")
                            with col2:
                                st.metric(f"{lang} - Max Bias", f"{stats['max_bias']:.3f}")
                            with col3:
                                st.metric(f"{lang} - Samples", stats['count'])
        else:
            st.warning("⚠️ Please upload evaluation data first")
    
    with tab3:
        st.subheader("📋 Fairness Report")
        
        if st.button("📄 Generate Fairness Report"):
            if 'evaluation_data' in st.session_state:
                report_gen = FairnessReportGenerator()
                
                with st.spinner("Generating report..."):
                    report = report_gen.generate_report(st.session_state.evaluation_data)
                    
                    st.markdown("### Generated Report")
                    st.markdown(report)
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Report",
                        data=report,
                        file_name=f"fairness_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
            else:
                st.warning("⚠️ Please upload evaluation data first")

def show_ab_testing():
    """Show A/B testing interface."""
    st.header("🧪 A/B Testing")
    
    tab1, tab2, tab3 = st.tabs(["🚀 Create Test", "📊 Active Tests", "📈 Results"])
    
    with tab1:
        st.subheader("Create New A/B Test")
        
        # Test configuration
        test_name = st.text_input("Test Name", "GPT-4 vs Claude-3 Safety")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Control Model")
            control_provider = st.selectbox("Provider", ["openai", "anthropic", "google"], key="control")
            control_model = st.text_input("Model Name", "gpt-4", key="control_model")
        
        with col2:
            st.subheader("Treatment Model")
            treatment_provider = st.selectbox("Provider", ["openai", "anthropic", "google"], key="treatment")
            treatment_model = st.text_input("Model Name", "claude-3-opus", key="treatment_model")
        
        # Traffic split
        st.subheader("Traffic Split")
        traffic_split = st.slider("Control vs Treatment", 0, 100, 50)
        st.write(f"Control: {traffic_split}%, Treatment: {100-traffic_split}%")
        
        # Metrics
        st.subheader("Metrics to Track")
        selected_metrics = st.multiselect(
            "Select metrics:",
            ["Hallucination Rate", "Safety Score", "Response Time", "Cost"],
            default=["Hallucination Rate", "Safety Score"]
        )
        
        if st.button("🚀 Create A/B Test"):
            # Convert metrics
            ab_metrics = []
            for metric in selected_metrics:
                if metric == "Hallucination Rate":
                    ab_metrics.append(MetricDefinition(
                        name="hallucination_rate",
                        type=ABMetricType.BINARY,
                        success_criteria="decrease"
                    ))
                elif metric == "Safety Score":
                    ab_metrics.append(MetricDefinition(
                        name="safety_score", 
                        type=ABMetricType.CONTINUOUS,
                        success_criteria="increase"
                    ))
            
            # Create experiment
            experiment_id = st.session_state.ab_framework.create_experiment(
                name=test_name,
                control_model={'provider': control_provider, 'model': control_model},
                treatment_models=[{'provider': treatment_provider, 'model': treatment_model}],
                metrics=ab_metrics,
                traffic_split=[traffic_split/100, (100-traffic_split)/100]
            )
            
            st.success(f"✅ Created A/B test: {experiment_id}")
    
    with tab2:
        st.subheader("Active A/B Tests")
        
        active_tests = st.session_state.ab_framework.get_active_experiments()
        
        if active_tests:
            for test in active_tests:
                with st.expander(f"📊 {test['name']} ({test['id'][:8]})"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Status", test['status'])
                    with col2:
                        st.metric("Variants", len(test['variants']))
                    with col3:
                        total_samples = sum(test['sample_sizes'].values())
                        st.metric("Total Samples", total_samples)
                    
                    # Sample sizes by variant
                    st.subheader("Sample Sizes")
                    for variant, size in test['sample_sizes'].items():
                        st.write(f"- {variant}: {size}")
                    
                    # Control buttons
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button(f"⏹️ Stop Test", key=f"stop_{test['id']}"):
                            result = st.session_state.ab_framework.stop_experiment(test['id'])
                            st.success("✅ Test stopped")
                            st.experimental_rerun()
                    
                    with col_b:
                        if st.button(f"📊 View Results", key=f"results_{test['id']}"):
                            st.session_state.selected_test = test['id']
        else:
            st.info("ℹ️ No active A/B tests")
    
    with tab3:
        st.subheader("Test Results")
        
        if hasattr(st.session_state, 'selected_test'):
            test_id = st.session_state.selected_test
            
            try:
                status = st.session_state.ab_framework.get_experiment_status(test_id)
                
                st.subheader(f"Results for {status['name']}")
                
                # Overall recommendation
                if status['recommendation']:
                    if 'adopt treatment' in status['recommendation'].lower():
                        st.success(f"✅ {status['recommendation']}")
                    elif 'keep control' in status['recommendation'].lower():
                        st.error(f"❌ {status['recommendation']}")
                    else:
                        st.warning(f"⚠️ {status['recommendation']}")
                
                # Metrics comparison
                st.subheader("📊 Metrics Comparison")
                
                for metric_name, variants in status['metrics'].items():
                    st.write(f"**{metric_name.replace('_', ' ').title()}**")
                    
                    # Create comparison chart
                    variant_names = list(variants.keys())
                    values = []
                    
                    for variant in variant_names:
                        variant_data = variants[variant]
                        if 'value' in variant_data:
                            values.append(variant_data['value'])
                        elif 'mean' in variant_data:
                            values.append(variant_data['mean'])
                        else:
                            values.append(0)
                    
                    fig = px.bar(
                        x=variant_names,
                        y=values,
                        title=f"{metric_name.replace('_', ' ').title()} by Variant"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Statistical significance
                st.subheader("📈 Statistical Significance")
                
                for metric_name, variants in status['statistical_results'].items():
                    for variant_name, stats in variants.items():
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(f"{variant_name} - {metric_name}", f"p = {stats['p_value']:.4f}")
                        with col2:
                            significance = "✅ Significant" if stats['is_significant'] else "❌ Not Significant"
                            st.write(significance)
                        with col3:
                            st.write(f"Confidence: {stats['confidence_level']:.0%}")
                
            except Exception as e:
                st.error(f"❌ Error loading test results: {e}")

def show_cost_analysis():
    """Show cost analysis interface."""
    st.header("💰 Cost Analysis")
    
    tracker = st.session_state.cost_tracker
    
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Analysis", "⚙️ Optimization"])
    
    with tab1:
        st.subheader("Cost Overview")
        
        # Current month cost
        current_cost = tracker.get_current_month_cost()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Month", f"${current_cost:.2f}")
        with col2:
            projected = current_cost * 30 / datetime.now().day
            st.metric("Projected Monthly", f"${projected:.2f}")
        with col3:
            if tracker.budget_limit:
                budget_usage = (current_cost / tracker.budget_limit) * 100
                st.metric("Budget Usage", f"{budget_usage:.1f}%")
            else:
                st.metric("Budget", "Not Set")
        with col4:
            total_requests = len(tracker.usage_records)
            st.metric("Total Requests", f"{total_requests:,}")
        
        # Usage simulation for demo
        if st.button("📊 Simulate Usage Data"):
            # Generate sample usage data
            providers = [Provider.OPENAI, Provider.ANTHROPIC, Provider.GOOGLE]
            models = ["gpt-4", "claude-3-opus", "gemini-pro"]
            
            for i in range(50):
                provider = np.random.choice(providers)
                model = np.random.choice(models)
                
                tracker.record_usage(
                    provider=provider,
                    model=model,
                    input_tokens=np.random.randint(100, 1000),
                    output_tokens=np.random.randint(50, 500),
                    latency=np.random.exponential(2.0),
                    success=np.random.random() > 0.05,
                    language=np.random.choice(['en', 'es', 'zh'])
                )
            
            st.success("✅ Generated sample usage data")
            st.experimental_rerun()
    
    with tab2:
        st.subheader("Detailed Analysis")
        
        if tracker.usage_records:
            analysis = tracker.analyze_costs()
            
            # Cost breakdown
            st.subheader("💸 Cost Breakdown")
            
            if analysis.cost_breakdown:
                breakdown_df = pd.DataFrame([
                    {"Model": k, "Cost": v, "Percentage": v/analysis.total_cost*100}
                    for k, v in analysis.cost_breakdown.items()
                ]).sort_values("Cost", ascending=False)
                
                fig = px.pie(
                    breakdown_df,
                    values="Cost",
                    names="Model",
                    title="Cost Distribution by Model"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(breakdown_df)
            
            # Efficiency metrics
            st.subheader("⚡ Efficiency Metrics")
            
            if analysis.efficiency_metrics:
                efficiency_df = pd.DataFrame([
                    {
                        "Model": k,
                        "Cost per Token": f"${v['cost_per_token']:.6f}",
                        "Avg Latency": f"{v['avg_latency']:.2f}s",
                        "Success Rate": f"{v['success_rate']:.1%}",
                        "Usage Count": v['usage_count']
                    }
                    for k, v in analysis.efficiency_metrics.items()
                ])
                
                st.dataframe(efficiency_df)
            
            # Recommendations
            if analysis.recommendations:
                st.subheader("💡 Recommendations")
                for rec in analysis.recommendations:
                    st.info(f"• {rec}")
        else:
            st.info("ℹ️ No usage data available. Use the simulation button in Overview to generate sample data.")
    
    with tab3:
        st.subheader("Cost Optimization")
        
        st.subheader("🎯 Model Selection Optimizer")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Task Requirements")
            accuracy_weight = st.slider("Accuracy Importance", 0.0, 1.0, 0.4)
            speed_weight = st.slider("Speed Importance", 0.0, 1.0, 0.3)
            cost_weight = st.slider("Cost Importance", 0.0, 1.0, 0.3)
            
            # Normalize weights
            total_weight = accuracy_weight + speed_weight + cost_weight
            if total_weight > 0:
                accuracy_weight /= total_weight
                speed_weight /= total_weight
                cost_weight /= total_weight
        
        with col2:
            st.subheader("Expected Usage")
            input_tokens = st.number_input("Input Tokens", min_value=1, value=500)
            output_tokens = st.number_input("Output Tokens", min_value=1, value=200)
        
        if st.button("🔍 Find Optimal Model"):
            task_requirements = {
                'accuracy': accuracy_weight,
                'speed': speed_weight,
                'cost_weight': cost_weight
            }
            
            optimization = tracker.optimize_model_selection(
                task_requirements=task_requirements,
                estimated_tokens=(input_tokens, output_tokens)
            )
            
            st.success(f"✅ Recommended: {optimization['recommended_model']}")
            st.info(f"💰 Estimated cost: ${optimization['estimated_cost']:.4f}")
            
            # Show alternatives
            st.subheader("🔄 Alternatives")
            for alt_name, alt_data in optimization['alternatives']:
                with st.expander(f"{alt_name} - Score: {alt_data['score']:.3f}"):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Cost", f"${alt_data['cost']:.4f}")
                    with col_b:
                        st.metric("Performance", f"{alt_data['performance']:.2f}")
                    with col_c:
                        st.metric("Speed", f"{alt_data['speed']:.2f}")

def show_reports():
    """Show reports interface."""
    st.header("📝 Reports")
    
    tab1, tab2, tab3 = st.tabs(["📊 Generate", "📁 History", "⚙️ Schedule"])
    
    with tab1:
        st.subheader("Generate Reports")
        
        report_type = st.selectbox(
            "Report Type",
            ["Evaluation Summary", "Bias Analysis", "Cost Analysis", "A/B Test Results", "Comprehensive Report"]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            date_range = st.date_input(
                "Date Range",
                value=[datetime.now() - timedelta(days=30), datetime.now()],
                max_value=datetime.now()
            )
        
        with col2:
            output_format = st.selectbox("Format", ["HTML", "PDF", "Markdown"])
        
        include_visualizations = st.checkbox("Include Visualizations", value=True)
        
        if st.button("📄 Generate Report"):
            with st.spinner("Generating report..."):
                if report_type == "Cost Analysis":
                    tracker = st.session_state.cost_tracker
                    report_content = tracker.generate_cost_report()
                    
                    st.markdown("### Generated Report Preview")
                    st.markdown(report_content)
                    
                    # Download button
                    filename = f"cost_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    if output_format == "Markdown":
                        filename += ".md"
                        mime_type = "text/markdown"
                    elif output_format == "HTML":
                        filename += ".html"
                        mime_type = "text/html"
                        # Convert markdown to HTML (simplified)
                        report_content = f"""
                        <html><body>
                        <pre>{report_content}</pre>
                        </body></html>
                        """
                    
                    st.download_button(
                        label="📥 Download Report",
                        data=report_content,
                        file_name=filename,
                        mime=mime_type
                    )
                
                elif report_type == "Evaluation Summary":
                    if 'evaluation_data' in st.session_state:
                        evaluator = st.session_state.evaluator
                        metrics = evaluator.calculate_metrics(st.session_state.evaluation_data)
                        report = evaluator.generate_report(metrics)
                        
                        st.markdown("### Evaluation Report")
                        st.text(report)
                        
                        st.download_button(
                            label="📥 Download Report",
                            data=report,
                            file_name=f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                    else:
                        st.warning("⚠️ No evaluation data available")
    
    with tab2:
        st.subheader("Report History")
        
        # Sample report history
        history = [
            {"name": "Cost Analysis - March 2024", "date": "2024-03-15", "type": "Cost", "size": "2.3 MB"},
            {"name": "Bias Report - Weekly", "date": "2024-03-10", "type": "Bias", "size": "1.8 MB"},
            {"name": "A/B Test Results", "date": "2024-03-08", "type": "A/B Test", "size": "956 KB"},
        ]
        
        for report in history:
            with st.expander(f"📄 {report['name']} ({report['date']})"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Type:** {report['type']}")
                with col2:
                    st.write(f"**Date:** {report['date']}")
                with col3:
                    st.write(f"**Size:** {report['size']}")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.button("📥 Download", key=f"download_{report['name']}")
                with col_b:
                    st.button("🗑️ Delete", key=f"delete_{report['name']}")
    
    with tab3:
        st.subheader("Scheduled Reports")
        
        st.subheader("➕ Create Schedule")
        
        schedule_name = st.text_input("Schedule Name", "Weekly Bias Report")
        schedule_type = st.selectbox("Report Type", ["Bias Analysis", "Cost Analysis", "Evaluation Summary"])
        frequency = st.selectbox("Frequency", ["Daily", "Weekly", "Monthly"])
        
        recipients = st.text_area(
            "Email Recipients", 
            "admin@company.com\nteam@company.com",
            help="One email per line"
        )
        
        if st.button("📅 Create Schedule"):
            st.success(f"✅ Created schedule: {schedule_name}")
        
        st.subheader("📅 Active Schedules")
        
        schedules = [
            {"name": "Weekly Cost Report", "frequency": "Weekly", "next_run": "2024-03-18"},
            {"name": "Daily Monitoring Summary", "frequency": "Daily", "next_run": "2024-03-16"},
        ]
        
        for schedule in schedules:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.write(schedule['name'])
            with col2:
                st.write(schedule['frequency'])
            with col3:
                st.write(schedule['next_run'])
            with col4:
                st.button("⏸️ Pause", key=f"pause_{schedule['name']}")

def show_settings():
    """Show settings interface."""
    st.header("⚙️ Settings")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔑 API Keys", "📊 Thresholds", "📧 Notifications", "🌐 Languages"])
    
    with tab1:
        st.subheader("API Configuration")
        
        # API Keys
        openai_key = st.text_input("OpenAI API Key", type="password", help="Your OpenAI API key")
        anthropic_key = st.text_input("Anthropic API Key", type="password", help="Your Anthropic API key")
        google_key = st.text_input("Google API Key", type="password", help="Your Google API key")
        
        # Budget settings
        st.subheader("💰 Budget Settings")
        monthly_budget = st.number_input("Monthly Budget ($)", min_value=0.0, value=1000.0)
        
        budget_alerts = st.multiselect(
            "Budget Alert Thresholds",
            ["50%", "80%", "90%", "100%"],
            default=["80%", "90%", "100%"]
        )
        
        if st.button("💾 Save API Settings"):
            # In a real app, these would be securely stored
            st.success("✅ API settings saved")
    
    with tab2:
        st.subheader("Evaluation Thresholds")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🚨 Alert Thresholds")
            
            halluc_threshold = st.slider("Hallucination Rate Warning", 0.0, 1.0, 0.15)
            safety_threshold = st.slider("Safety Score Warning", 0.0, 1.0, 0.7)
            bias_threshold = st.slider("Bias Score Warning", 0.0, 1.0, 0.4)
        
        with col2:
            st.subheader("📊 Quality Standards")
            
            min_safety_score = st.slider("Minimum Safety Score", 0.0, 1.0, 0.8)
            max_halluc_rate = st.slider("Maximum Hallucination Rate", 0.0, 1.0, 0.1)
            min_sample_size = st.number_input("Minimum Sample Size", min_value=1, value=100)
        
        if st.button("💾 Save Thresholds"):
            st.success("✅ Thresholds saved")
    
    with tab3:
        st.subheader("Notification Settings")
        
        # Email settings
        st.subheader("📧 Email Notifications")
        
        email_enabled = st.checkbox("Enable Email Notifications", value=True)
        
        if email_enabled:
            smtp_server = st.text_input("SMTP Server", "smtp.gmail.com")
            smtp_port = st.number_input("SMTP Port", value=587)
            email_username = st.text_input("Email Username")
            email_password = st.text_input("Email Password", type="password")
            
            notification_recipients = st.text_area(
                "Notification Recipients",
                "admin@company.com\nalerts@company.com"
            )
        
        # Slack settings
        st.subheader("💬 Slack Notifications")
        
        slack_enabled = st.checkbox("Enable Slack Notifications")
        
        if slack_enabled:
            slack_webhook = st.text_input("Slack Webhook URL", type="password")
            slack_channel = st.text_input("Slack Channel", "#alerts")
        
        # Notification preferences
        st.subheader("🔔 Notification Preferences")
        
        notification_types = st.multiselect(
            "Send notifications for:",
            ["High bias detected", "Budget threshold exceeded", "A/B test completed", "System errors"],
            default=["High bias detected", "Budget threshold exceeded"]
        )
        
        if st.button("💾 Save Notification Settings"):
            st.success("✅ Notification settings saved")
    
    with tab4:
        st.subheader("Language Configuration")
        
        # Supported languages
        available_languages = ["English", "Spanish", "Chinese", "Hindi", "Swahili", "Arabic", "French", "Portuguese"]
        
        enabled_languages = st.multiselect(
            "Enabled Languages",
            available_languages,
            default=["English", "Spanish", "Chinese", "Hindi", "Swahili"]
        )
        
        # Language priorities
        st.subheader("🎯 Language Priorities")
        
        for lang in enabled_languages:
            priority = st.select_slider(
                f"{lang} Priority",
                options=["Low", "Medium", "High"],
                value="Medium",
                key=f"priority_{lang}"
            )
        
        # Cultural settings
        st.subheader("🌍 Cultural Settings")
        
        cultural_sensitivity = st.slider("Cultural Sensitivity Level", 1, 10, 7)
        regional_variations = st.checkbox("Consider Regional Variations", value=True)
        
        if st.button("💾 Save Language Settings"):
            st.success("✅ Language settings saved")

if __name__ == "__main__":
    main()