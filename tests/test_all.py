"""
Unit tests for multilingual alignment evaluation modules.
"""

import unittest
import tempfile
import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.evaluate import MultilingualEvaluator
from scripts.data_processing import DataValidator, DataPreprocessor, DataAugmenter
from scripts.prompt_manager import PromptTemplate, MultilingualPromptManager
from scripts.model_adapters import ModelConfig, ModelResponse


class TestMultilingualEvaluator(unittest.TestCase):
    """Test cases for MultilingualEvaluator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = MultilingualEvaluator()
        self.test_data = pd.DataFrame({
            'text': ['Response 1', 'Response 2', 'Response 3'],
            'hallucinated': [False, True, False],
            'safety_score': [0.9, 0.6, 0.8],
            'language': ['en', 'sw', 'en'],
            'model': ['gpt-4', 'gpt-4', 'claude-3']
        })
        
    def test_load_data(self):
        """Test data loading."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            self.test_data.to_json(f, orient='records', lines=True)
            temp_path = f.name
            
        try:
            df = self.evaluator.load_data(temp_path)
            self.assertEqual(len(df), 3)
            self.assertIn('hallucinated', df.columns)
        finally:
            os.unlink(temp_path)
            
    def test_calculate_metrics(self):
        """Test metric calculation."""
        metrics = self.evaluator.calculate_metrics(self.test_data)
        
        # Check basic metrics
        self.assertIn('hallucination_rate', metrics)
        self.assertIn('average_safety_score', metrics)
        self.assertIn('total_samples', metrics)
        
        # Check values
        self.assertAlmostEqual(metrics['hallucination_rate'], 1/3, places=3)
        self.assertAlmostEqual(metrics['average_safety_score'], 0.767, places=2)
        self.assertEqual(metrics['total_samples'], 3)
        
        # Check language metrics
        self.assertIn('per_language_metrics', metrics)
        self.assertEqual(len(metrics['per_language_metrics']), 2)
        
    def test_generate_report(self):
        """Test report generation."""
        metrics = self.evaluator.calculate_metrics(self.test_data)
        report = self.evaluator.generate_report(metrics)
        
        self.assertIn('MULTILINGUAL ALIGNMENT EVALUATION REPORT', report)
        self.assertIn('Hallucination Rate:', report)
        self.assertIn('Average Safety Score:', report)
        
    def test_export_metrics(self):
        """Test metrics export."""
        metrics = self.evaluator.calculate_metrics(self.test_data)
        
        # Test JSON export
        json_output = self.evaluator.export_metrics(metrics, format='json')
        parsed = json.loads(json_output)
        self.assertEqual(parsed['total_samples'], 3)
        
        # Test CSV export
        csv_output = self.evaluator.export_metrics(metrics, format='csv')
        self.assertIn('total_samples', csv_output)


class TestDataProcessing(unittest.TestCase):
    """Test cases for data processing modules."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = DataValidator()
        self.preprocessor = DataPreprocessor()
        self.augmenter = DataAugmenter()
        
        self.test_df = pd.DataFrame({
            'text': ['Valid text', '', 'Another text'],
            'hallucinated': [True, False, 'true'],  # Mixed types
            'safety_score': [0.8, 1.5, -0.1],  # Out of range values
            'language': ['en', 'unknown', 'sw']
        })
        
    def test_validate_dataset(self):
        """Test dataset validation."""
        validated_df, report = self.validator.validate_dataset(
            self.test_df.copy(), strict=False
        )
        
        # Check that validation occurred
        self.assertIn('validation_summary', report)
        self.assertIn('data_quality_metrics', report)
        
        # Check that boolean conversion happened
        self.assertTrue(pd.api.types.is_bool_dtype(validated_df['hallucinated']))
        
    def test_validate_score_ranges(self):
        """Test score range validation."""
        df = self.test_df.copy()
        validated_df = self.validator._validate_score_ranges(df)
        
        # Check scores are clipped to valid range
        self.assertTrue((validated_df['safety_score'] >= 0).all())
        self.assertTrue((validated_df['safety_score'] <= 1).all())
        
    def test_preprocess_data(self):
        """Test data preprocessing."""
        processed_df = self.preprocessor.preprocess(self.test_df.copy())
        
        # Check that empty rows are handled
        self.assertLessEqual(len(processed_df), len(self.test_df))
        
        # Check derived features
        if 'text' in processed_df.columns:
            self.assertIn('text_length', processed_df.columns)
            self.assertIn('word_count', processed_df.columns)
            
    def test_augment_dataset(self):
        """Test data augmentation."""
        augmented_df = self.augmenter.augment_dataset(
            self.test_df[['text', 'safety_score']].dropna(),
            augmentation_factor=0.5
        )
        
        # Check that augmentation added rows
        self.assertGreater(len(augmented_df), len(self.test_df))
        
        # Check augmentation markers
        self.assertIn('is_augmented', augmented_df.columns)
        self.assertTrue(augmented_df['is_augmented'].any())


class TestPromptManager(unittest.TestCase):
    """Test cases for prompt management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = MultilingualPromptManager()
        
        # Create test template
        self.test_template = PromptTemplate(
            template_id="test_template",
            category="test_category",
            templates={
                "en": "Test {variable} in English",
                "sw": "Jaribio {variable} kwa Kiswahili"
            },
            variables={"variable": "default"}
        )
        
    def test_add_template(self):
        """Test adding templates."""
        initial_count = len(self.manager.templates)
        self.manager.add_template(self.test_template)
        
        self.assertEqual(len(self.manager.templates), initial_count + 1)
        self.assertIn("test_template", self.manager.templates)
        
    def test_render_template(self):
        """Test template rendering."""
        self.manager.add_template(self.test_template)
        
        # Test English rendering
        rendered = self.manager.render_template(
            "test_template", "en", variable="example"
        )
        self.assertEqual(rendered, "Test example in English")
        
        # Test Swahili rendering
        rendered = self.manager.render_template(
            "test_template", "sw", variable="mfano"
        )
        self.assertEqual(rendered, "Jaribio mfano kwa Kiswahili")
        
    def test_language_fallback(self):
        """Test language fallback mechanism."""
        self.manager.add_template(self.test_template)
        
        # Test non-existent language falls back to English
        rendered = self.test_template.render("fr", variable="test")
        self.assertIn("English", rendered)
        
    def test_generate_test_suite(self):
        """Test test suite generation."""
        test_suite = self.manager.generate_test_suite(
            languages=['en', 'sw'],
            categories=['safety_testing'],
            samples_per_template=2
        )
        
        self.assertIsInstance(test_suite, list)
        if test_suite:
            self.assertIn('test_id', test_suite[0])
            self.assertIn('prompt', test_suite[0])
            self.assertIn('language', test_suite[0])
            
    def test_get_statistics(self):
        """Test statistics generation."""
        stats = self.manager.get_statistics()
        
        self.assertIn('total_templates', stats)
        self.assertIn('categories', stats)
        self.assertIn('languages', stats)
        self.assertGreater(stats['total_templates'], 0)


class TestModelAdapters(unittest.TestCase):
    """Test cases for model adapters."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ModelConfig(
            model_name="test-model",
            max_tokens=100,
            temperature=0.7
        )
        
    def test_model_response(self):
        """Test ModelResponse dataclass."""
        response = ModelResponse(
            text="Test response",
            model="test-model",
            usage={"total_tokens": 10},
            raw_response={},
            latency=0.5
        )
        
        self.assertEqual(response.text, "Test response")
        self.assertTrue(response.success)
        self.assertIsNone(response.error)
        
        # Test to_dict conversion
        response_dict = response.to_dict()
        self.assertIn('text', response_dict)
        self.assertIn('latency', response_dict)
        
    def test_model_config(self):
        """Test ModelConfig dataclass."""
        self.assertEqual(self.config.model_name, "test-model")
        self.assertEqual(self.config.max_tokens, 100)
        self.assertIsNotNone(self.config.additional_params)
        
    def test_validate_prompt(self):
        """Test prompt validation."""
        # This would require mocking the adapter
        # For now, just test the configuration
        self.assertIsInstance(self.config.temperature, float)
        self.assertGreater(self.config.max_tokens, 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the full pipeline."""
    
    def test_full_evaluation_pipeline(self):
        """Test the complete evaluation pipeline."""
        # Create test data
        test_data = pd.DataFrame({
            'text': ['Safe response', 'Unsafe content', 'Normal text'],
            'hallucinated': [False, True, False],
            'safety_score': [0.9, 0.3, 0.7],
            'language': ['en', 'en', 'sw'],
            'model': ['gpt-4', 'gpt-4', 'gpt-4']
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            test_data.to_json(f, orient='records', lines=True)
            temp_path = f.name
            
        try:
            # Initialize components
            evaluator = MultilingualEvaluator()
            
            # Load and evaluate
            df = evaluator.load_data(temp_path)
            metrics = evaluator.calculate_metrics(df)
            
            # Generate report
            report = evaluator.generate_report(metrics)
            
            # Verify results
            self.assertIsInstance(metrics, dict)
            self.assertIsInstance(report, str)
            self.assertIn('hallucination_rate', metrics)
            
        finally:
            os.unlink(temp_path)
            
    def test_preprocessing_pipeline(self):
        """Test data preprocessing pipeline."""
        # Create test data with issues
        test_data = pd.DataFrame({
            'text': ['  Valid text  ', '', 'Another text!!!'],
            'hallucinated': ['true', 'false', True],
            'safety_score': [0.8, 1.5, -0.1],
            'language': ['en', 'unknown', 'sw']
        })
        
        # Validate and preprocess
        validator = DataValidator()
        preprocessor = DataPreprocessor()
        
        validated_df, report = validator.validate_dataset(test_data, strict=False)
        processed_df = preprocessor.preprocess(validated_df)
        
        # Check results
        self.assertLessEqual(len(processed_df), len(test_data))
        self.assertTrue((processed_df['safety_score'] >= 0).all())
        self.assertTrue((processed_df['safety_score'] <= 1).all())


# Test runner
if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)