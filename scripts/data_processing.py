"""
Data preprocessing and validation module for multilingual alignment evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import json
import re
from pathlib import Path
import logging
from datetime import datetime
import hashlib
from collections import defaultdict

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates and preprocesses evaluation data."""
    
    def __init__(self, config: Dict = None):
        """Initialize validator with configuration."""
        self.config = config or {}
        self.validation_report = defaultdict(list)
        
    def validate_dataset(self, df: pd.DataFrame, strict: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        Comprehensive dataset validation.
        
        Args:
            df: Input dataframe
            strict: If True, raises errors on validation failures
            
        Returns:
            Tuple of (validated_df, validation_report)
        """
        logger.info(f"Starting validation for dataset with {len(df)} records")
        
        # Check required fields
        required_fields = self.config.get('data', {}).get('validation', {}).get(
            'required_fields', ['text', 'hallucinated', 'safety_score']
        )
        self._check_required_fields(df, required_fields, strict)
        
        # Validate data types
        df = self._validate_data_types(df)
        
        # Validate score ranges
        df = self._validate_score_ranges(df)
        
        # Validate text content
        df = self._validate_text_content(df)
        
        # Validate language codes if present
        if 'language' in df.columns:
            df = self._validate_language_codes(df)
        
        # Check for duplicates
        df = self._check_duplicates(df)
        
        # Generate validation report
        report = self._generate_validation_report(df)
        
        logger.info(f"Validation complete. {len(df)} records passed validation")
        
        return df, report
    
    def _check_required_fields(self, df: pd.DataFrame, required_fields: List[str], 
                              strict: bool) -> None:
        """Check for required fields in dataframe."""
        missing_fields = set(required_fields) - set(df.columns)
        if missing_fields:
            msg = f"Missing required fields: {missing_fields}"
            self.validation_report['errors'].append(msg)
            if strict:
                raise ValueError(msg)
            else:
                logger.warning(msg)
    
    def _validate_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and convert data types."""
        # Validate boolean fields
        if 'hallucinated' in df.columns:
            if not pd.api.types.is_bool_dtype(df['hallucinated']):
                logger.warning("Converting 'hallucinated' to boolean")
                df['hallucinated'] = df['hallucinated'].astype(bool)
                self.validation_report['warnings'].append(
                    "Converted 'hallucinated' column to boolean"
                )
        
        # Validate numeric fields
        numeric_fields = ['safety_score', 'confidence_score', 'fluency_score']
        for field in numeric_fields:
            if field in df.columns:
                if not pd.api.types.is_numeric_dtype(df[field]):
                    try:
                        df[field] = pd.to_numeric(df[field], errors='coerce')
                        invalid_count = df[field].isna().sum()
                        if invalid_count > 0:
                            self.validation_report['warnings'].append(
                                f"Found {invalid_count} invalid values in '{field}'"
                            )
                    except Exception as e:
                        self.validation_report['errors'].append(
                            f"Could not convert '{field}' to numeric: {e}"
                        )
        
        return df
    
    def _validate_score_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate that scores are within expected ranges."""
        score_range = self.config.get('data', {}).get('validation', {}).get(
            'score_range', [0, 1]
        )
        min_score, max_score = score_range
        
        score_fields = ['safety_score', 'confidence_score', 'fluency_score']
        for field in score_fields:
            if field in df.columns:
                out_of_range = df[(df[field] < min_score) | (df[field] > max_score)]
                if len(out_of_range) > 0:
                    self.validation_report['warnings'].append(
                        f"{len(out_of_range)} values in '{field}' outside range {score_range}"
                    )
                    # Clip values to range
                    df[field] = df[field].clip(min_score, max_score)
        
        return df
    
    def _validate_text_content(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate text content fields."""
        text_fields = ['text', 'prompt', 'response']
        
        for field in text_fields:
            if field in df.columns:
                # Check for empty strings
                empty_count = (df[field].str.strip() == '').sum()
                if empty_count > 0:
                    self.validation_report['warnings'].append(
                        f"Found {empty_count} empty values in '{field}'"
                    )
                
                # Check for very short texts (potential issues)
                short_texts = (df[field].str.len() < 10).sum()
                if short_texts > 0:
                    self.validation_report['info'].append(
                        f"Found {short_texts} very short texts (<10 chars) in '{field}'"
                    )
                
                # Check for suspicious patterns (e.g., repeated characters)
                pattern = r'(.)\1{10,}'  # Same character repeated 10+ times
                suspicious = df[field].str.contains(pattern, regex=True, na=False).sum()
                if suspicious > 0:
                    self.validation_report['warnings'].append(
                        f"Found {suspicious} texts with suspicious patterns in '{field}'"
                    )
        
        return df
    
    def _validate_language_codes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate language codes against supported languages."""
        supported_langs = self.config.get('evaluation', {}).get('languages', {}).get(
            'supported', ['en', 'sw', 'hi', 'id', 'zh', 'es', 'ar', 'fr']
        )
        
        if 'language' in df.columns:
            invalid_langs = df[~df['language'].isin(supported_langs)]['language'].unique()
            if len(invalid_langs) > 0:
                self.validation_report['warnings'].append(
                    f"Found unsupported language codes: {list(invalid_langs)}"
                )
                # Optionally map or filter invalid languages
                df = df[df['language'].isin(supported_langs)]
        
        return df
    
    def _check_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check for and handle duplicate entries."""
        # Check for exact duplicates
        exact_duplicates = df.duplicated().sum()
        if exact_duplicates > 0:
            self.validation_report['warnings'].append(
                f"Found {exact_duplicates} exact duplicate rows"
            )
            df = df.drop_duplicates()
        
        # Check for duplicate texts (potential data leakage)
        if 'text' in df.columns:
            text_duplicates = df.duplicated(subset=['text']).sum()
            if text_duplicates > 0:
                self.validation_report['info'].append(
                    f"Found {text_duplicates} duplicate text entries"
                )
        
        return df
    
    def _generate_validation_report(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive validation report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_records': len(df),
            'validation_summary': dict(self.validation_report),
            'data_quality_metrics': {
                'completeness': self._calculate_completeness(df),
                'consistency': self._calculate_consistency(df),
                'validity': self._calculate_validity(df)
            },
            'field_statistics': self._calculate_field_statistics(df)
        }
        
        return report
    
    def _calculate_completeness(self, df: pd.DataFrame) -> float:
        """Calculate data completeness score."""
        total_cells = df.size
        non_null_cells = df.count().sum()
        return non_null_cells / total_cells if total_cells > 0 else 0
    
    def _calculate_consistency(self, df: pd.DataFrame) -> float:
        """Calculate data consistency score."""
        # Simple consistency check based on expected patterns
        consistency_score = 1.0
        
        # Check if safety scores align with hallucination flags
        if 'hallucinated' in df.columns and 'safety_score' in df.columns:
            # Generally, hallucinated content should have lower safety scores
            halluc_safety_mean = df[df['hallucinated'] == True]['safety_score'].mean()
            non_halluc_safety_mean = df[df['hallucinated'] == False]['safety_score'].mean()
            
            if halluc_safety_mean > non_halluc_safety_mean:
                consistency_score *= 0.8
                self.validation_report['warnings'].append(
                    "Inconsistency detected: hallucinated content has higher safety scores"
                )
        
        return consistency_score
    
    def _calculate_validity(self, df: pd.DataFrame) -> float:
        """Calculate data validity score."""
        validity_issues = 0
        total_checks = 0
        
        # Check each validation warning/error
        for category in ['errors', 'warnings']:
            validity_issues += len(self.validation_report[category])
            total_checks += 10  # Approximate number of validation checks
        
        return max(0, 1 - (validity_issues / total_checks))
    
    def _calculate_field_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate statistics for each field."""
        stats = {}
        
        for col in df.columns:
            col_stats = {
                'dtype': str(df[col].dtype),
                'null_count': int(df[col].isna().sum()),
                'null_percentage': float(df[col].isna().mean() * 100),
                'unique_count': int(df[col].nunique())
            }
            
            # Add numeric statistics
            if pd.api.types.is_numeric_dtype(df[col]):
                col_stats.update({
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'median': float(df[col].median())
                })
            
            # Add text statistics
            elif pd.api.types.is_string_dtype(df[col]):
                text_lengths = df[col].str.len()
                col_stats.update({
                    'avg_length': float(text_lengths.mean()),
                    'min_length': int(text_lengths.min()),
                    'max_length': int(text_lengths.max())
                })
            
            stats[col] = col_stats
        
        return stats


class DataPreprocessor:
    """Preprocesses evaluation data for analysis."""
    
    def __init__(self, config: Dict = None):
        """Initialize preprocessor with configuration."""
        self.config = config or {}
        
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all preprocessing steps."""
        logger.info("Starting data preprocessing")
        
        # Remove empty rows if configured
        if self.config.get('data', {}).get('preprocessing', {}).get('remove_empty', True):
            df = self._remove_empty_rows(df)
        
        # Normalize scores if configured
        if self.config.get('data', {}).get('preprocessing', {}).get('normalize_scores', True):
            df = self._normalize_scores(df)
        
        # Handle missing values
        handle_missing = self.config.get('data', {}).get('preprocessing', {}).get(
            'handle_missing', 'drop'
        )
        df = self._handle_missing_values(df, strategy=handle_missing)
        
        # Clean text fields
        df = self._clean_text_fields(df)
        
        # Add derived features
        df = self._add_derived_features(df)
        
        # Sort by timestamp if available
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
        
        logger.info(f"Preprocessing complete. {len(df)} records processed")
        
        return df
    
    def _remove_empty_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with all empty values."""
        before_count = len(df)
        df = df.dropna(how='all')
        removed = before_count - len(df)
        
        if removed > 0:
            logger.info(f"Removed {removed} completely empty rows")
        
        return df
    
    def _normalize_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize score fields to [0, 1] range."""
        score_fields = ['safety_score', 'confidence_score', 'fluency_score']
        
        for field in score_fields:
            if field in df.columns:
                # Check if normalization is needed
                if df[field].max() > 1 or df[field].min() < 0:
                    # Simple min-max normalization
                    df[field] = (df[field] - df[field].min()) / (
                        df[field].max() - df[field].min()
                    )
                    logger.info(f"Normalized '{field}' to [0, 1] range")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
        """Handle missing values based on strategy."""
        if strategy == 'drop':
            # Drop rows with missing required fields
            required_fields = self.config.get('data', {}).get('validation', {}).get(
                'required_fields', ['text', 'hallucinated', 'safety_score']
            )
            df = df.dropna(subset=required_fields)
        
        elif strategy == 'fill':
            # Fill missing values with defaults
            fill_values = {
                'safety_score': 0.5,
                'confidence_score': 0.5,
                'hallucinated': False,
                'language': 'unknown',
                'model': 'unknown'
            }
            df = df.fillna(fill_values)
        
        elif strategy == 'impute':
            # More sophisticated imputation could be added here
            pass
        
        return df
    
    def _clean_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize text fields."""
        text_fields = ['text', 'prompt', 'response']
        
        for field in text_fields:
            if field in df.columns:
                # Strip whitespace
                df[field] = df[field].str.strip()
                
                # Remove excessive whitespace
                df[field] = df[field].str.replace(r'\s+', ' ', regex=True)
                
                # Remove control characters
                df[field] = df[field].str.replace(r'[\x00-\x1F\x7F-\x9F]', '', regex=True)
        
        return df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add useful derived features for analysis."""
        # Text length features
        if 'text' in df.columns:
            df['text_length'] = df['text'].str.len()
            df['word_count'] = df['text'].str.split().str.len()
        
        # Safety categories
        if 'safety_score' in df.columns:
            df['safety_category'] = pd.cut(
                df['safety_score'],
                bins=[0, 0.3, 0.5, 0.8, 1.0],
                labels=['critical', 'low', 'medium', 'high']
            )
        
        # Language family (if language column exists)
        if 'language' in df.columns:
            language_families = {
                'en': 'germanic',
                'sw': 'bantu',
                'hi': 'indo-aryan',
                'id': 'austronesian',
                'zh': 'sino-tibetan',
                'es': 'romance',
                'ar': 'semitic',
                'fr': 'romance'
            }
            df['language_family'] = df['language'].map(language_families)
        
        # Add hash for deduplication tracking
        if 'text' in df.columns:
            df['text_hash'] = df['text'].apply(
                lambda x: hashlib.md5(x.encode()).hexdigest()[:8]
            )
        
        return df


class DataAugmenter:
    """Augments evaluation data for more robust testing."""
    
    def __init__(self):
        """Initialize data augmenter."""
        self.augmentation_stats = defaultdict(int)
    
    def augment_dataset(self, df: pd.DataFrame, augmentation_factor: float = 0.2) -> pd.DataFrame:
        """
        Augment dataset with synthetic variations.
        
        Args:
            df: Original dataframe
            augmentation_factor: Fraction of data to augment (0.2 = 20% new data)
            
        Returns:
            Augmented dataframe
        """
        logger.info(f"Starting data augmentation with factor {augmentation_factor}")
        
        augmented_rows = []
        num_to_augment = int(len(df) * augmentation_factor)
        
        for _ in range(num_to_augment):
            # Select random row to augment
            row = df.sample(n=1).iloc[0].copy()
            
            # Apply random augmentation
            augmented_row = self._augment_row(row)
            augmented_rows.append(augmented_row)
        
        # Combine original and augmented data
        augmented_df = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)
        
        logger.info(f"Augmentation complete. Added {len(augmented_rows)} synthetic samples")
        logger.info(f"Augmentation statistics: {dict(self.augmentation_stats)}")
        
        return augmented_df
    
    def _augment_row(self, row: pd.Series) -> pd.Series:
        """Apply augmentation to a single row."""
        augmented = row.copy()
        
        # Choose augmentation strategy
        strategies = ['paraphrase', 'noise', 'translation_back', 'typos']
        strategy = np.random.choice(strategies)
        
        if strategy == 'paraphrase' and 'text' in row:
            augmented['text'] = self._paraphrase_text(row['text'])
            augmented['augmentation_type'] = 'paraphrase'
            
        elif strategy == 'noise' and 'safety_score' in row:
            # Add small noise to scores
            noise = np.random.normal(0, 0.05)
            augmented['safety_score'] = np.clip(row['safety_score'] + noise, 0, 1)
            augmented['augmentation_type'] = 'score_noise'
            
        elif strategy == 'translation_back' and 'text' in row:
            # Simulate back-translation effects
            augmented['text'] = self._simulate_back_translation(row['text'])
            augmented['augmentation_type'] = 'back_translation'
            
        elif strategy == 'typos' and 'text' in row:
            augmented['text'] = self._add_typos(row['text'])
            augmented['augmentation_type'] = 'typos'
        
        self.augmentation_stats[strategy] += 1
        augmented['is_augmented'] = True
        
        return augmented
    
    def _paraphrase_text(self, text: str) -> str:
        """Simple paraphrasing by word shuffling (placeholder for real paraphrasing)."""
        words = text.split()
        if len(words) > 3:
            # Shuffle middle words while keeping first and last
            middle = words[1:-1]
            np.random.shuffle(middle)
            return f"{words[0]} {' '.join(middle)} {words[-1]}"
        return text
    
    def _simulate_back_translation(self, text: str) -> str:
        """Simulate effects of back-translation."""
        # Simple simulation: slight word changes
        replacements = {
            'the': 'a',
            'is': 'was',
            'are': 'were',
            'good': 'nice',
            'bad': 'poor'
        }
        
        for old, new in replacements.items():
            if old in text.lower():
                text = text.replace(old, new)
                break
        
        return text
    
    def _add_typos(self, text: str, typo_rate: float = 0.02) -> str:
        """Add realistic typos to text."""
        if len(text) < 10:
            return text
        
        text_list = list(text)
        num_typos = max(1, int(len(text) * typo_rate))
        
        for _ in range(num_typos):
            pos = np.random.randint(1, len(text_list) - 1)
            typo_type = np.random.choice(['swap', 'delete', 'duplicate'])
            
            if typo_type == 'swap' and pos < len(text_list) - 1:
                text_list[pos], text_list[pos + 1] = text_list[pos + 1], text_list[pos]
            elif typo_type == 'delete':
                text_list[pos] = ''
            elif typo_type == 'duplicate':
                text_list[pos] = text_list[pos] * 2
        
        return ''.join(text_list)


# Utility functions
def load_and_validate_data(file_path: str, config_path: Optional[str] = None,
                          strict: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to load, validate, and preprocess data.
    
    Args:
        file_path: Path to data file
        config_path: Optional path to configuration file
        strict: Whether to enforce strict validation
        
    Returns:
        Tuple of (processed_dataframe, validation_report)
    """
    # Load configuration
    config = {}
    if config_path:
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    # Load data
    df = pd.read_json(file_path, lines=True)
    
    # Validate
    validator = DataValidator(config)
    df, validation_report = validator.validate_dataset(df, strict=strict)
    
    # Preprocess
    preprocessor = DataPreprocessor(config)
    df = preprocessor.preprocess(df)
    
    return df, validation_report


def generate_sample_data(num_samples: int = 100, 
                        languages: List[str] = None,
                        models: List[str] = None) -> pd.DataFrame:
    """Generate sample data for testing."""
    if languages is None:
        languages = ['en', 'sw', 'hi', 'id']
    if models is None:
        models = ['gpt-4', 'claude-3', 'llama-2']
    
    data = []
    for i in range(num_samples):
        record = {
            'text': f"Sample response {i} with some content that could be evaluated.",
            'hallucinated': np.random.choice([True, False], p=[0.2, 0.8]),
            'safety_score': np.random.beta(7, 3),  # Skewed towards higher scores
            'language': np.random.choice(languages),
            'model': np.random.choice(models),
            'timestamp': datetime.now().isoformat(),
            'prompt_type': np.random.choice(['safety', 'factual', 'creative']),
            'confidence_score': np.random.beta(5, 2)
        }
        data.append(record)
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Data preprocessing and validation")
    parser.add_argument('--generate', type=int, help='Generate N sample records')
    parser.add_argument('--validate', type=str, help='Validate data file')
    parser.add_argument('--augment', type=str, help='Augment data file')
    parser.add_argument('--output', type=str, help='Output file path')
    
    args = parser.parse_args()
    
    if args.generate:
        df = generate_sample_data(args.generate)
        output_path = args.output or 'generated_sample_data.jsonl'
        df.to_json(output_path, orient='records', lines=True)
        print(f"Generated {len(df)} samples to {output_path}")
    
    elif args.validate:
        df, report = load_and_validate_data(args.validate, strict=False)
        print("Validation Report:")
        print(json.dumps(report, indent=2))
        
        if args.output:
            df.to_json(args.output, orient='records', lines=True)
            print(f"Validated data saved to {args.output}")
    
    elif args.augment:
        df = pd.read_json(args.augment, lines=True)
        augmenter = DataAugmenter()
        augmented_df = augmenter.augment_dataset(df)
        
        output_path = args.output or 'augmented_data.jsonl'
        augmented_df.to_json(output_path, orient='records', lines=True)
        print(f"Augmented data saved to {output_path}")