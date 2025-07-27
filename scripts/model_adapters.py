"""
Model interface adapters for various LLMs (Large Language Models).
Provides unified interface for different model providers.
"""

import os
import json
import time
import asyncio
from typing import Dict, List, Optional, Union, Any, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
import tiktoken
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


@dataclass
class ModelResponse:
    """Standardized model response format."""
    text: str
    model: str
    usage: Dict[str, int]
    raw_response: Dict
    latency: float
    success: bool = True
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert response to dictionary."""
        return {
            'text': self.text,
            'model': self.model,
            'usage': self.usage,
            'latency': self.latency,
            'success': self.success,
            'error': self.error
        }


@dataclass
class ModelConfig:
    """Configuration for a model."""
    model_name: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    max_tokens: int = 500
    temperature: float = 0.7
    top_p: float = 0.9
    timeout: int = 30
    max_retries: int = 3
    additional_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_params is None:
            self.additional_params = {}


class BaseModelAdapter(ABC):
    """Base class for model adapters."""
    
    def __init__(self, config: ModelConfig):
        """Initialize the adapter with configuration."""
        self.config = config
        self.client = None
        self._setup_client()
        
    @abstractmethod
    def _setup_client(self):
        """Setup the API client."""
        pass
    
    @abstractmethod
    async def generate_async(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate response asynchronously."""
        pass
    
    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate response synchronously."""
        return asyncio.run(self.generate_async(prompt, **kwargs))
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        pass
    
    def validate_prompt(self, prompt: str) -> bool:
        """Validate prompt before sending to model."""
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")
        
        token_count = self.count_tokens(prompt)
        if token_count > 4000:  # Conservative limit
            logger.warning(f"Prompt has {token_count} tokens, may be too long")
            
        return True


class OpenAIAdapter(BaseModelAdapter):
    """Adapter for OpenAI models (GPT-4, GPT-3.5-turbo, etc.)."""
    
    def _setup_client(self):
        """Setup OpenAI client."""
        api_key = self.config.api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not provided")
            
        self.api_key = api_key
        self.api_base = self.config.api_base or "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def generate_async(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate response from OpenAI model."""
        self.validate_prompt(prompt)
        
        # Merge kwargs with config
        params = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get('max_tokens', self.config.max_tokens),
            "temperature": kwargs.get('temperature', self.config.temperature),
            "top_p": kwargs.get('top_p', self.config.top_p),
            **self.config.additional_params
        }
        
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_base}/chat/completions",
                    headers=self.headers,
                    json=params,
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                
            data = response.json()
            
            return ModelResponse(
                text=data['choices'][0]['message']['content'],
                model=data['model'],
                usage=data['usage'],
                raw_response=data,
                latency=time.time() - start_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return ModelResponse(
                text="",
                model=self.config.model_name,
                usage={},
                raw_response={},
                latency=time.time() - start_time,
                success=False,
                error=str(e)
            )
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        try:
            encoding = tiktoken.encoding_for_model(self.config.model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
            
        return len(encoding.encode(text))


class AnthropicAdapter(BaseModelAdapter):
    """Adapter for Anthropic models (Claude)."""
    
    def _setup_client(self):
        """Setup Anthropic client."""
        api_key = self.config.api_key or os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("Anthropic API key not provided")
            
        self.api_key = api_key
        self.api_base = self.config.api_base or "https://api.anthropic.com/v1"
        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def generate_async(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate response from Anthropic model."""
        self.validate_prompt(prompt)
        
        params = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get('max_tokens', self.config.max_tokens),
            "temperature": kwargs.get('temperature', self.config.temperature),
            **self.config.additional_params
        }
        
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_base}/messages",
                    headers=self.headers,
                    json=params,
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                
            data = response.json()
            
            return ModelResponse(
                text=data['content'][0]['text'],
                model=data['model'],
                usage={
                    'prompt_tokens': data['usage']['input_tokens'],
                    'completion_tokens': data['usage']['output_tokens'],
                    'total_tokens': data['usage']['input_tokens'] + data['usage']['output_tokens']
                },
                raw_response=data,
                latency=time.time() - start_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return ModelResponse(
                text="",
                model=self.config.model_name,
                usage={},
                raw_response={},
                latency=time.time() - start_time,
                success=False,
                error=str(e)
            )
    
    def count_tokens(self, text: str) -> int:
        """Approximate token count for Claude models."""
        # Rough approximation: 1 token ≈ 4 characters
        return len(text) // 4


class GoogleAdapter(BaseModelAdapter):
    """Adapter for Google models (PaLM, Gemini)."""
    
    def _setup_client(self):
        """Setup Google client."""
        api_key = self.config.api_key or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("Google API key not provided")
            
        self.api_key = api_key
        self.api_base = self.config.api_base or "https://generativelanguage.googleapis.com/v1"
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def generate_async(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate response from Google model."""
        self.validate_prompt(prompt)
        
        endpoint = f"{self.api_base}/models/{self.config.model_name}:generateContent"
        
        params = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": kwargs.get('temperature', self.config.temperature),
                "topP": kwargs.get('top_p', self.config.top_p),
                "maxOutputTokens": kwargs.get('max_tokens', self.config.max_tokens),
            }
        }
        
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{endpoint}?key={self.api_key}",
                    json=params,
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                
            data = response.json()
            
            text = data['candidates'][0]['content']['parts'][0]['text']
            
            return ModelResponse(
                text=text,
                model=self.config.model_name,
                usage={
                    'prompt_tokens': len(prompt.split()),  # Approximation
                    'completion_tokens': len(text.split()),
                    'total_tokens': len(prompt.split()) + len(text.split())
                },
                raw_response=data,
                latency=time.time() - start_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Google API error: {e}")
            return ModelResponse(
                text="",
                model=self.config.model_name,
                usage={},
                raw_response={},
                latency=time.time() - start_time,
                success=False,
                error=str(e)
            )
    
    def count_tokens(self, text: str) -> int:
        """Approximate token count for Google models."""
        # Rough approximation based on words
        return len(text.split())


class HuggingFaceAdapter(BaseModelAdapter):
    """Adapter for HuggingFace models."""
    
    def _setup_client(self):
        """Setup HuggingFace client."""
        api_key = self.config.api_key or os.getenv('HUGGINGFACE_API_KEY')
        if not api_key:
            raise ValueError("HuggingFace API key not provided")
            
        self.api_key = api_key
        self.api_base = self.config.api_base or "https://api-inference.huggingface.co/models"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def generate_async(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate response from HuggingFace model."""
        self.validate_prompt(prompt)
        
        params = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": kwargs.get('max_tokens', self.config.max_tokens),
                "temperature": kwargs.get('temperature', self.config.temperature),
                "top_p": kwargs.get('top_p', self.config.top_p),
                "do_sample": True,
                **self.config.additional_params
            }
        }
        
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_base}/{self.config.model_name}",
                    headers=self.headers,
                    json=params,
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                
            data = response.json()
            
            # Extract text from response
            if isinstance(data, list) and len(data) > 0:
                text = data[0].get('generated_text', '')
                # Remove the prompt from the beginning if present
                if text.startswith(prompt):
                    text = text[len(prompt):].strip()
            else:
                text = str(data)
            
            return ModelResponse(
                text=text,
                model=self.config.model_name,
                usage={
                    'prompt_tokens': len(prompt.split()),
                    'completion_tokens': len(text.split()),
                    'total_tokens': len(prompt.split()) + len(text.split())
                },
                raw_response=data,
                latency=time.time() - start_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"HuggingFace API error: {e}")
            return ModelResponse(
                text="",
                model=self.config.model_name,
                usage={},
                raw_response={},
                latency=time.time() - start_time,
                success=False,
                error=str(e)
            )
    
    def count_tokens(self, text: str) -> int:
        """Approximate token count."""
        # Rough approximation
        return len(text.split())


class LocalModelAdapter(BaseModelAdapter):
    """Adapter for local models (using transformers library)."""
    
    def _setup_client(self):
        """Setup local model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map="auto",
                torch_dtype="auto"
            )
        except ImportError:
            raise ImportError("transformers library required for local models")
        except Exception as e:
            raise RuntimeError(f"Failed to load local model: {e}")
    
    async def generate_async(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate response from local model."""
        import torch
        
        self.validate_prompt(prompt)
        start_time = time.time()
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                    temperature=kwargs.get('temperature', self.config.temperature),
                    top_p=kwargs.get('top_p', self.config.top_p),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove prompt from output
            if text.startswith(prompt):
                text = text[len(prompt):].strip()
            
            return ModelResponse(
                text=text,
                model=self.config.model_name,
                usage={
                    'prompt_tokens': len(inputs['input_ids'][0]),
                    'completion_tokens': len(outputs[0]) - len(inputs['input_ids'][0]),
                    'total_tokens': len(outputs[0])
                },
                raw_response={'outputs': outputs},
                latency=time.time() - start_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Local model error: {e}")
            return ModelResponse(
                text="",
                model=self.config.model_name,
                usage={},
                raw_response={},
                latency=time.time() - start_time,
                success=False,
                error=str(e)
            )
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using the model's tokenizer."""
        return len(self.tokenizer.encode(text))


class ModelFactory:
    """Factory for creating model adapters."""
    
    _adapters = {
        'openai': OpenAIAdapter,
        'anthropic': AnthropicAdapter,
        'google': GoogleAdapter,
        'huggingface': HuggingFaceAdapter,
        'local': LocalModelAdapter
    }
    
    @classmethod
    def create(cls, provider: str, config: ModelConfig) -> BaseModelAdapter:
        """
        Create a model adapter.
        
        Args:
            provider: Model provider ('openai', 'anthropic', etc.)
            config: Model configuration
            
        Returns:
            Model adapter instance
        """
        if provider not in cls._adapters:
            raise ValueError(f"Unknown provider: {provider}")
            
        return cls._adapters[provider](config)
    
    @classmethod
    def register_adapter(cls, provider: str, adapter_class: type):
        """Register a new adapter class."""
        cls._adapters[provider] = adapter_class


class MultiModelEvaluator:
    """Evaluates prompts across multiple models."""
    
    def __init__(self, models: Dict[str, BaseModelAdapter]):
        """
        Initialize with multiple model adapters.
        
        Args:
            models: Dictionary of model_name -> adapter
        """
        self.models = models
        
    async def evaluate_prompt_async(self, prompt: str, **kwargs) -> Dict[str, ModelResponse]:
        """Evaluate a prompt across all models asynchronously."""
        tasks = {
            name: adapter.generate_async(prompt, **kwargs)
            for name, adapter in self.models.items()
        }
        
        results = {}
        for name, task in tasks.items():
            try:
                results[name] = await task
            except Exception as e:
                logger.error(f"Error evaluating {name}: {e}")
                results[name] = ModelResponse(
                    text="",
                    model=name,
                    usage={},
                    raw_response={},
                    latency=0,
                    success=False,
                    error=str(e)
                )
                
        return results
    
    def evaluate_prompt(self, prompt: str, **kwargs) -> Dict[str, ModelResponse]:
        """Evaluate a prompt across all models synchronously."""
        return asyncio.run(self.evaluate_prompt_async(prompt, **kwargs))
    
    def evaluate_dataset(self, prompts: List[str], 
                        parallel: bool = True,
                        max_workers: int = 5) -> List[Dict[str, ModelResponse]]:
        """
        Evaluate multiple prompts across all models.
        
        Args:
            prompts: List of prompts to evaluate
            parallel: Whether to process prompts in parallel
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of evaluation results
        """
        if parallel:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self.evaluate_prompt, prompt)
                    for prompt in prompts
                ]
                
                results = []
                for future in as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        logger.error(f"Error in parallel evaluation: {e}")
                        
                return results
        else:
            return [self.evaluate_prompt(prompt) for prompt in prompts]
    
    def compare_responses(self, results: Dict[str, ModelResponse]) -> Dict[str, Any]:
        """Compare responses from different models."""
        comparison = {
            'consensus': None,
            'disagreement_score': 0,
            'latency_comparison': {},
            'token_efficiency': {},
            'success_rate': 0
        }
        
        # Extract successful responses
        successful_responses = {
            name: resp for name, resp in results.items() 
            if resp.success
        }
        
        comparison['success_rate'] = len(successful_responses) / len(results)
        
        if not successful_responses:
            return comparison
        
        # Compare latencies
        comparison['latency_comparison'] = {
            name: resp.latency for name, resp in successful_responses.items()
        }
        
        # Compare token efficiency
        comparison['token_efficiency'] = {
            name: resp.usage.get('total_tokens', 0) / max(len(resp.text), 1)
            for name, resp in successful_responses.items()
        }
        
        # Simple consensus check (could be more sophisticated)
        responses = [resp.text for resp in successful_responses.values()]
        if len(set(responses)) == 1:
            comparison['consensus'] = True
            comparison['disagreement_score'] = 0
        else:
            comparison['consensus'] = False
            # Simple disagreement score based on response diversity
            comparison['disagreement_score'] = len(set(responses)) / len(responses)
        
        return comparison


# Example usage
if __name__ == "__main__":
    # Configure models
    models = {
        'gpt-4': ModelFactory.create('openai', ModelConfig(
            model_name='gpt-4',
            temperature=0.7
        )),
        'claude-3': ModelFactory.create('anthropic', ModelConfig(
            model_name='claude-3-opus-20240229',
            temperature=0.7
        ))
    }
    
    # Create evaluator
    evaluator = MultiModelEvaluator(models)
    
    # Test prompt
    test_prompt = "What is the capital of France?"
    
    # Evaluate
    results = evaluator.evaluate_prompt(test_prompt)
    
    # Print results
    for model_name, response in results.items():
        print(f"\n{model_name}:")
        print(f"  Response: {response.text[:100]}...")
        print(f"  Latency: {response.latency:.2f}s")
        print(f"  Success: {response.success}")
        
    # Compare responses
    comparison = evaluator.compare_responses(results)
    print(f"\nComparison: {json.dumps(comparison, indent=2)}")