"""
CLI tool for multilingual alignment evaluation.
Provides a user-friendly command-line interface for all evaluation tasks.
"""

import click
import json
import pandas as pd
from pathlib import Path
import sys
from typing import Optional, List
import logging
from datetime import datetime
import os

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.evaluate import MultilingualEvaluator
from scripts.data_processing import (
    DataValidator, DataPreprocessor, DataAugmenter, 
    generate_sample_data, load_and_validate_data
)
from scripts.prompt_manager import MultilingualPromptManager
from scripts.model_adapters import ModelFactory, ModelConfig, MultiModelEvaluator
from scripts.visualization import EvaluationVisualizer, ReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--config', '-c', type=click.Path(exists=True), 
              help='Path to configuration file')
@click.pass_context
def cli(ctx, verbose, config):
    """Multilingual Alignment Evaluation Tool
    
    A comprehensive toolkit for evaluating language model alignment
    across multiple languages, with focus on safety and accuracy.
    """
    ctx.ensure_object(dict)
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Load configuration if provided
    if config:
        with open(config, 'r') as f:
            ctx.obj['config'] = json.load(f)
    else:
        ctx.obj['config'] = {}
        
    click.echo(click.style("🌍 Multilingual Alignment Evaluation Tool", fg='blue', bold=True))


@cli.command()
@click.argument('data_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), 
              help='Output file for evaluation results')
@click.option('--report', '-r', type=click.Path(), 
              help='Generate report file')
@click.option('--format', '-f', type=click.Choice(['json', 'csv', 'html', 'pdf']), 
              default='json', help='Output format')
@click.option('--visualize', '-v', is_flag=True, 
              help='Generate visualizations')
@click.pass_context
def evaluate(ctx, data_file, output, report, format, visualize):
    """Evaluate alignment metrics on a dataset.
    
    DATA_FILE: Path to the evaluation data file (JSONL format)
    """
    click.echo(f"📊 Evaluating data from: {data_file}")
    
    try:
        # Initialize evaluator
        config_path = ctx.obj.get('config_path')
        evaluator = MultilingualEvaluator(config_path=config_path)
        
        # Load and evaluate data
        with click.progressbar(length=3, label='Processing') as bar:
            # Load data
            data = evaluator.load_data(data_file)
            bar.update(1)
            
            # Calculate metrics
            metrics = evaluator.calculate_metrics(data)
            bar.update(1)
            
            # Generate report
            report_text = evaluator.generate_report(metrics, output_path=report)
            bar.update(1)
        
        # Display results
        click.echo("\n" + click.style("Evaluation Results:", fg='green', bold=True))
        click.echo(f"  Total Samples: {metrics['total_samples']}")
        click.echo(f"  Hallucination Rate: {metrics['hallucination_rate']:.2%}")
        click.echo(f"  Average Safety Score: {metrics['average_safety_score']:.3f}")
        
        if 'languages_evaluated' in metrics:
            click.echo(f"  Languages Evaluated: {metrics['languages_evaluated']}")
            
        # Save results if output specified
        if output:
            evaluator.export_metrics(metrics, format=format, output_path=output)
            click.echo(f"\n💾 Results saved to: {output}")
            
        # Generate visualizations if requested
        if visualize:
            visualizer = EvaluationVisualizer()
            vis_path = Path(output).parent / f"{Path(output).stem}_dashboard.html"
            visualizer.create_overview_dashboard(metrics, output_path=str(vis_path))
            click.echo(f"📈 Dashboard saved to: {vis_path}")
            
        # Generate report if requested
        if report:
            report_gen = ReportGenerator()
            if format == 'html':
                report_path = report_gen.generate_html_report(metrics, df=data, output_path=report)
            else:
                report_path = report_gen.generate_pdf_report(metrics, df=data, output_path=report)
            click.echo(f"📄 Report saved to: {report_path}")
            
    except Exception as e:
        click.echo(click.style(f"❌ Error: {e}", fg='red'))
        raise click.Abort()


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), required=True,
              help='Output file for processed data')
@click.option('--validate-only', is_flag=True, 
              help='Only validate without preprocessing')
@click.option('--augment', '-a', type=float, 
              help='Augmentation factor (e.g., 0.2 for 20% augmentation)')
@click.option('--strict', is_flag=True, 
              help='Use strict validation (fail on errors)')
def preprocess(input_file, output, validate_only, augment, strict):
    """Preprocess and validate evaluation data.
    
    INPUT_FILE: Path to the raw data file
    """
    click.echo(f"🔧 Processing data from: {input_file}")
    
    try:
        # Load data
        df = pd.read_json(input_file, lines=True)
        click.echo(f"  Loaded {len(df)} records")
        
        # Validate
        validator = DataValidator()
        validated_df, validation_report = validator.validate_dataset(df, strict=strict)
        
        # Display validation results
        click.echo("\n" + click.style("Validation Report:", fg='yellow', bold=True))
        if validation_report['validation_summary'].get('errors'):
            for error in validation_report['validation_summary']['errors']:
                click.echo(click.style(f"  ❌ {error}", fg='red'))
                
        if validation_report['validation_summary'].get('warnings'):
            for warning in validation_report['validation_summary']['warnings']:
                click.echo(click.style(f"  ⚠️  {warning}", fg='yellow'))
                
        click.echo(f"\n  Data Quality Score: {validation_report['data_quality_metrics']['validity']:.2f}")
        
        if validate_only:
            click.echo("\n✅ Validation complete (no preprocessing applied)")
            return
            
        # Preprocess
        preprocessor = DataPreprocessor()
        processed_df = preprocessor.preprocess(validated_df)
        click.echo(f"\n  Preprocessed to {len(processed_df)} records")
        
        # Augment if requested
        if augment:
            augmenter = DataAugmenter()
            processed_df = augmenter.augment_dataset(processed_df, augmentation_factor=augment)
            click.echo(f"  Augmented to {len(processed_df)} records")
            
        # Save
        processed_df.to_json(output, orient='records', lines=True)
        click.echo(f"\n💾 Processed data saved to: {output}")
        
    except Exception as e:
        click.echo(click.style(f"❌ Error: {e}", fg='red'))
        raise click.Abort()


@cli.command()
@click.option('--num-samples', '-n', type=int, default=100,
              help='Number of samples to generate')
@click.option('--languages', '-l', multiple=True, 
              default=['en', 'sw', 'hi', 'id'],
              help='Languages to include')
@click.option('--models', '-m', multiple=True,
              default=['gpt-4', 'claude-3', 'llama-2'],
              help='Models to include')
@click.option('--output', '-o', type=click.Path(), required=True,
              help='Output file path')
def generate(num_samples, languages, models, output):
    """Generate sample evaluation data for testing."""
    click.echo(f"🎲 Generating {num_samples} sample records...")
    
    try:
        # Generate data
        df = generate_sample_data(
            num_samples=num_samples,
            languages=list(languages),
            models=list(models)
        )
        
        # Save
        df.to_json(output, orient='records', lines=True)
        
        # Display summary
        click.echo("\n" + click.style("Generated Data Summary:", fg='green', bold=True))
        click.echo(f"  Total Records: {len(df)}")
        click.echo(f"  Languages: {', '.join(df['language'].unique())}")
        click.echo(f"  Models: {', '.join(df['model'].unique())}")
        click.echo(f"  Hallucination Rate: {df['hallucinated'].mean():.2%}")
        click.echo(f"  Avg Safety Score: {df['safety_score'].mean():.3f}")
        click.echo(f"\n💾 Data saved to: {output}")
        
    except Exception as e:
        click.echo(click.style(f"❌ Error: {e}", fg='red'))
        raise click.Abort()


@cli.command()
@click.option('--languages', '-l', multiple=True, required=True,
              help='Languages to test')
@click.option('--categories', '-c', multiple=True,
              help='Prompt categories to include')
@click.option('--output', '-o', type=click.Path(), required=True,
              help='Output file for test suite')
@click.option('--samples', '-s', type=int, default=3,
              help='Samples per template')
def prompts(languages, categories, output, samples):
    """Generate multilingual prompt test suite."""
    click.echo("📝 Generating prompt test suite...")
    
    try:
        # Initialize prompt manager
        manager = MultilingualPromptManager()
        
        # Get statistics
        stats = manager.get_statistics()
        click.echo(f"\n  Available Templates: {stats['total_templates']}")
        click.echo(f"  Categories: {', '.join(stats['categories'].keys())}")
        
        # Generate test suite
        test_suite = manager.generate_test_suite(
            languages=list(languages),
            categories=list(categories) if categories else None,
            samples_per_template=samples
        )
        
        # Save
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(test_suite, f, indent=2, ensure_ascii=False)
            
        click.echo(f"\n  Generated {len(test_suite)} test prompts")
        click.echo(f"💾 Test suite saved to: {output}")
        
    except Exception as e:
        click.echo(click.style(f"❌ Error: {e}", fg='red'))
        raise click.Abort()


@cli.command()
@click.argument('prompt_file', type=click.Path(exists=True))
@click.option('--models', '-m', multiple=True, required=True,
              help='Models to evaluate (format: provider:model_name)')
@click.option('--output', '-o', type=click.Path(), required=True,
              help='Output file for results')
@click.option('--parallel', '-p', is_flag=True,
              help='Run evaluations in parallel')
def test_models(prompt_file, models, output, parallel):
    """Test multiple models with a prompt suite.
    
    PROMPT_FILE: JSON file containing test prompts
    """
    click.echo("🤖 Testing models with prompt suite...")
    
    try:
        # Load prompts
        with open(prompt_file, 'r') as f:
            if prompt_file.endswith('.json'):
                prompts_data = json.load(f)
                prompts = [p['prompt'] for p in prompts_data]
            else:
                prompts = [line.strip() for line in f if line.strip()]
                
        click.echo(f"  Loaded {len(prompts)} prompts")
        
        # Initialize models
        model_adapters = {}
        for model_spec in models:
            provider, model_name = model_spec.split(':', 1)
            config = ModelConfig(model_name=model_name)
            adapter = ModelFactory.create(provider, config)
            model_adapters[model_spec] = adapter
            
        click.echo(f"  Initialized {len(model_adapters)} models")
        
        # Create evaluator
        evaluator = MultiModelEvaluator(model_adapters)
        
        # Evaluate
        click.echo("\n⏳ Running evaluations...")
        results = []
        
        with click.progressbar(prompts, label='Testing') as bar:
            for prompt in bar:
                if parallel:
                    result = evaluator.evaluate_prompt(prompt)
                else:
                    result = evaluator.evaluate_prompt(prompt)
                    
                results.append({
                    'prompt': prompt,
                    'responses': {k: v.to_dict() for k, v in result.items()}
                })
                
        # Save results
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)
            
        click.echo(f"\n💾 Results saved to: {output}")
        
        # Display summary
        success_rates = {}
        for model in model_adapters:
            successes = sum(1 for r in results 
                          if r['responses'].get(model, {}).get('success', False))
            success_rates[model] = successes / len(results)
            
        click.echo("\n" + click.style("Success Rates:", fg='green', bold=True))
        for model, rate in success_rates.items():
            click.echo(f"  {model}: {rate:.1%}")
            
    except Exception as e:
        click.echo(click.style(f"❌ Error: {e}", fg='red'))
        raise click.Abort()


@cli.command()
@click.option('--input', '-i', type=click.Path(exists=True), required=True,
              help='Evaluation results file')
@click.option('--type', '-t', 
              type=click.Choice(['dashboard', 'heatmap', 'trends', 'comparison']),
              default='dashboard', help='Visualization type')
@click.option('--output', '-o', type=click.Path(), required=True,
              help='Output file path')
@click.option('--groupby', '-g', type=str, default='language',
              help='Group by field for comparisons')
def visualize(input, type, output, groupby):
    """Create visualizations from evaluation results."""
    click.echo(f"📊 Creating {type} visualization...")
    
    try:
        # Load data
        if input.endswith('.json'):
            with open(input, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    metrics = data
                    df = None
                else:
                    # Convert to dataframe
                    df = pd.DataFrame(data)
                    metrics = None
        else:
            df = pd.read_json(input, lines=True)
            metrics = None
            
        # Initialize visualizer
        visualizer = EvaluationVisualizer()
        
        # Create visualization based on type
        if type == 'dashboard':
            if metrics is None:
                # Calculate metrics from dataframe
                evaluator = MultilingualEvaluator()
                metrics = evaluator.calculate_metrics(df)
            fig = visualizer.create_overview_dashboard(metrics, output_path=output)
            
        elif type == 'heatmap' and df is not None:
            fig = visualizer.plot_language_heatmap(df, save_path=output)
            
        elif type == 'trends' and df is not None:
            fig = visualizer.plot_temporal_trends(df, save_path=output)
            
        elif type == 'comparison' and df is not None:
            figs = visualizer.create_comparison_plots(df, groupby=groupby, 
                                                    save_dir=Path(output).parent)
            
        click.echo(f"✅ Visualization saved to: {output}")
        
    except Exception as e:
        click.echo(click.style(f"❌ Error: {e}", fg='red'))
        raise click.Abort()


@cli.command()
def info():
    """Display information about the evaluation system."""
    click.echo("\n" + click.style("Multilingual Alignment Evaluation System", 
                                  fg='blue', bold=True))
    click.echo("=" * 50)
    click.echo("\n📋 " + click.style("Components:", bold=True))
    click.echo("  • Enhanced evaluation metrics")
    click.echo("  • Data validation and preprocessing")
    click.echo("  • Multilingual prompt templates")
    click.echo("  • Model adapter interfaces")
    click.echo("  • Visualization and reporting")
    click.echo("  • Comprehensive CLI tools")
    
    click.echo("\n🌍 " + click.style("Supported Languages:", bold=True))
    languages = {
        'en': 'English',
        'sw': 'Swahili',
        'hi': 'Hindi',
        'id': 'Bahasa Indonesia',
        'zh': 'Chinese',
        'es': 'Spanish',
        'ar': 'Arabic',
        'fr': 'French'
    }
    for code, name in languages.items():
        click.echo(f"  • {code}: {name}")
        
    click.echo("\n🤖 " + click.style("Supported Model Providers:", bold=True))
    providers = ['openai', 'anthropic', 'google', 'huggingface', 'local']
    for provider in providers:
        click.echo(f"  • {provider}")
        
    click.echo("\n📊 " + click.style("Evaluation Metrics:", bold=True))
    metrics = [
        'Hallucination Rate',
        'Safety Score',
        'Language Consistency',
        'Cultural Sensitivity',
        'Token Efficiency'
    ]
    for metric in metrics:
        click.echo(f"  • {metric}")
        
    click.echo("\n💡 " + click.style("Example Commands:", bold=True))
    examples = [
        "mleval evaluate data/responses.jsonl -o results.json -v",
        "mleval generate -n 1000 -l en sw hi -o test_data.jsonl",
        "mleval preprocess raw_data.jsonl -o clean_data.jsonl --augment 0.2",
        "mleval prompts -l en sw -c safety_testing -o prompts.json",
        "mleval visualize -i results.json -t dashboard -o dashboard.html"
    ]
    for example in examples:
        click.echo(f"  $ {example}")
        
    click.echo("\n📚 For more information, visit the project repository")
    click.echo("   or use --help with any command\n")


# Entry point
if __name__ == '__main__':
    cli()