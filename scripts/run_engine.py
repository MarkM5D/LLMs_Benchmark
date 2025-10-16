#!/usr/bin/env python3
"""
Engine Adapter - Unified interface for all LLM engines
Provides standardized methods for loading models and generating text across vLLM, SGLang, and TensorRT-LLM
"""

import time
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import logging

# Engine-specific imports (with fallbacks)
try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = SamplingParams = None

try:
    import sglang as sgl
    from sglang import Runtime, set_default_backend
except ImportError:
    sgl = Runtime = set_default_backend = None

try:
    import tensorrt_llm
    from tensorrt_llm.runtime import ModelRunner, SamplingConfig
except ImportError:
    tensorrt_llm = ModelRunner = SamplingConfig = None

class BaseEngineAdapter(ABC):
    """Base class for all engine adapters"""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        self.model_name = model_name
        self.config = config
        self.model = None
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the engine and load the model"""
        pass
    
    @abstractmethod
    def generate(self, prompts: List[str], sampling_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate text for given prompts"""
        pass
    
    @abstractmethod
    def cleanup(self):
        """Cleanup resources"""
        pass
    
    def is_available(self) -> bool:
        """Check if the engine dependencies are available"""
        return True
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get engine information"""
        return {
            "engine_name": self.__class__.__name__.replace("Adapter", "").lower(),
            "model_name": self.model_name,
            "config": self.config
        }

class VLLMAdapter(BaseEngineAdapter):
    """vLLM Engine Adapter"""
    
    def is_available(self) -> bool:
        return LLM is not None and SamplingParams is not None
    
    def initialize(self) -> bool:
        """Initialize vLLM model"""
        if not self.is_available():
            self.logger.error("vLLM is not installed")
            return False
            
        try:
            self.logger.info(f"Initializing vLLM model: {self.model_name}")
            
            vllm_config = {
                "model": self.model_name,
                "tensor_parallel_size": self.config.get("tensor_parallel_size", 1),
                "gpu_memory_utilization": self.config.get("gpu_memory_utilization", 0.85),
                "max_model_len": self.config.get("max_model_len", 4096),
                "enforce_eager": self.config.get("enforce_eager", False),
                "disable_log_stats": self.config.get("disable_log_stats", True)
            }
            
            # Add optional parameters
            if "max_num_seqs" in self.config:
                vllm_config["max_num_seqs"] = self.config["max_num_seqs"]
                
            self.model = LLM(**vllm_config)
            self.logger.info("vLLM model initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vLLM: {e}")
            return False
    
    def generate(self, prompts: List[str], sampling_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate text using vLLM"""
        if self.model is None:
            raise RuntimeError("Model not initialized")
            
        # Convert sampling parameters
        vllm_sampling = SamplingParams(
            temperature=sampling_params.get("temperature", 0.8),
            top_p=sampling_params.get("top_p", 0.95),
            top_k=sampling_params.get("top_k", -1),
            max_tokens=sampling_params.get("max_tokens", 512),
            repetition_penalty=sampling_params.get("repetition_penalty", 1.0),
            stop=sampling_params.get("stop", None)
        )
        
        start_time = time.perf_counter()
        outputs = self.model.generate(prompts, vllm_sampling)
        total_time = time.perf_counter() - start_time
        
        # Convert outputs to standard format
        results = []
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            token_count = len(output.outputs[0].token_ids)
            
            results.append({
                "prompt": prompts[i],
                "generated_text": generated_text,
                "tokens_generated": token_count,
                "generation_time_ms": (total_time / len(prompts)) * 1000,
                "tokens_per_second": token_count / (total_time / len(prompts)) if total_time > 0 else 0,
                "engine_specific": {
                    "finish_reason": output.outputs[0].finish_reason,
                    "logprobs": getattr(output.outputs[0], 'logprobs', None)
                }
            })
        
        return results
    
    def cleanup(self):
        """Cleanup vLLM resources"""
        if self.model is not None:
            # vLLM doesn't have explicit cleanup
            self.model = None

class SGLangAdapter(BaseEngineAdapter):
    """SGLang Engine Adapter"""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self.runtime = None
    
    def is_available(self) -> bool:
        return sgl is not None and Runtime is not None
    
    def initialize(self) -> bool:
        """Initialize SGLang runtime"""
        if not self.is_available():
            self.logger.error("SGLang is not installed")
            return False
            
        try:
            self.logger.info(f"Initializing SGLang runtime: {self.model_name}")
            
            self.runtime = Runtime(
                model_path=self.model_name,
                tp_size=self.config.get("tp_size", 1),
                mem_fraction_static=self.config.get("mem_fraction_static", 0.85),
                max_running_requests=self.config.get("max_running_requests", 100),
                context_length=self.config.get("context_length", 4096)
            )
            set_default_backend(self.runtime)
            
            self.logger.info("SGLang runtime initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize SGLang: {e}")
            return False
    
    def generate(self, prompts: List[str], sampling_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate text using SGLang"""
        if self.runtime is None:
            raise RuntimeError("Runtime not initialized")
        
        # Define SGLang generation function
        @sgl.function
        def generate_text(s, prompt, max_tokens, temperature, top_p, frequency_penalty):
            s += sgl.user(prompt)
            s += sgl.assistant(sgl.gen(
                "response", 
                max_tokens=max_tokens, 
                temperature=temperature, 
                top_p=top_p,
                frequency_penalty=frequency_penalty
            ))
        
        results = []
        for prompt in prompts:
            start_time = time.perf_counter()
            
            state = generate_text.run(
                prompt=prompt,
                max_tokens=sampling_params.get("max_tokens", 512),
                temperature=sampling_params.get("temperature", 0.8),
                top_p=sampling_params.get("top_p", 0.95),
                frequency_penalty=sampling_params.get("repetition_penalty", 1.1) - 1.0  # Convert to frequency penalty
            )
            
            generation_time = (time.perf_counter() - start_time) * 1000
            
            # Extract response
            response = getattr(state, 'response', '') if hasattr(state, 'response') else ''
            token_count = len(response.split()) + len(response) // 4  # Rough estimate
            
            results.append({
                "prompt": prompt,
                "generated_text": response,
                "tokens_generated": token_count,
                "generation_time_ms": generation_time,
                "tokens_per_second": token_count / (generation_time / 1000) if generation_time > 0 else 0,
                "engine_specific": {
                    "state": str(type(state)),
                    "radix_cache_hit": None  # SGLang specific info
                }
            })
        
        return results
    
    def cleanup(self):
        """Cleanup SGLang resources"""
        if self.runtime is not None:
            self.runtime.shutdown()
            self.runtime = None

class TensorRTLLMAdapter(BaseEngineAdapter):
    """TensorRT-LLM Engine Adapter"""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self.runner = None
    
    def is_available(self) -> bool:
        return tensorrt_llm is not None and ModelRunner is not None
    
    def initialize(self) -> bool:
        """Initialize TensorRT-LLM model runner"""
        if not self.is_available():
            self.logger.error("TensorRT-LLM is not installed")
            return False
            
        try:
            engine_path = self.config.get("engine_path")
            if not engine_path:
                engine_path = f"./engines/{self.model_name}/1-gpu"
                
            self.logger.info(f"Initializing TensorRT-LLM from: {engine_path}")
            
            self.runner = ModelRunner.from_dir(
                engine_dir=engine_path,
                lora_dir=self.config.get("lora_dir"),
                rank=self.config.get("rank", 0),
                world_size=self.config.get("world_size", 1),
                max_batch_size=self.config.get("max_batch_size", 32),
                max_input_len=self.config.get("max_input_len", 2048),
                max_output_len=self.config.get("max_output_len", 512),
                max_beam_width=1
            )
            
            self.logger.info("TensorRT-LLM runner initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize TensorRT-LLM: {e}")
            return False
    
    def generate(self, prompts: List[str], sampling_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate text using TensorRT-LLM"""
        if self.runner is None:
            raise RuntimeError("Runner not initialized")
        
        # Configure sampling
        sampling_config = SamplingConfig(
            end_id=50256,  # Adjust based on tokenizer
            pad_id=50256,
            num_beams=1,
            temperature=sampling_params.get("temperature", 0.8),
            top_k=sampling_params.get("top_k", 50),
            top_p=sampling_params.get("top_p", 0.95),
            repetition_penalty=sampling_params.get("repetition_penalty", 1.1),
            length_penalty=1.0
        )
        
        start_time = time.perf_counter()
        
        outputs = self.runner.generate(
            batch_input_ids=None,
            batch_input_texts=prompts,
            max_new_tokens=sampling_params.get("max_tokens", 512),
            sampling_config=sampling_config,
            output_sequence_lengths=True,
            return_dict=True
        )
        
        total_time = time.perf_counter() - start_time
        
        # Convert outputs to standard format
        results = []
        output_texts = outputs.get('output_texts', [''] * len(prompts))
        output_ids = outputs.get('output_ids', [])
        
        for i, (prompt, output_text) in enumerate(zip(prompts, output_texts)):
            # Calculate token count
            if i < len(output_ids) and hasattr(output_ids[i], 'shape'):
                token_count = output_ids[i].shape[-1]
            else:
                token_count = len(output_text.split()) + len(output_text) // 4
            
            results.append({
                "prompt": prompt,
                "generated_text": output_text,
                "tokens_generated": token_count,
                "generation_time_ms": (total_time / len(prompts)) * 1000,
                "tokens_per_second": token_count / (total_time / len(prompts)) if total_time > 0 else 0,
                "engine_specific": {
                    "tensorrt_optimized": True,
                    "sequence_length": len(output_text) if output_text else 0
                }
            })
        
        return results
    
    def cleanup(self):
        """Cleanup TensorRT-LLM resources"""
        if self.runner is not None:
            # TensorRT-LLM doesn't have explicit cleanup in ModelRunner
            self.runner = None

class EngineFactory:
    """Factory for creating engine adapters"""
    
    ADAPTERS = {
        "vllm": VLLMAdapter,
        "sglang": SGLangAdapter,
        "tensorrt": TensorRTLLMAdapter,
        "tensorrt-llm": TensorRTLLMAdapter
    }
    
    @classmethod
    def create_adapter(cls, engine_name: str, model_name: str, config: Dict[str, Any]) -> BaseEngineAdapter:
        """Create an engine adapter"""
        if engine_name not in cls.ADAPTERS:
            raise ValueError(f"Unknown engine: {engine_name}. Available: {list(cls.ADAPTERS.keys())}")
        
        adapter_class = cls.ADAPTERS[engine_name]
        return adapter_class(model_name, config)
    
    @classmethod
    def list_available_engines(cls) -> List[str]:
        """List available engines with their installation status"""
        available = []
        for name, adapter_class in cls.ADAPTERS.items():
            # Create a dummy adapter to check availability
            dummy_adapter = adapter_class("dummy", {})
            if dummy_adapter.is_available():
                available.append(name)
        return available

def main():
    """Test the engine adapters"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Engine Adapters")
    parser.add_argument("--engine", required=True, choices=list(EngineFactory.ADAPTERS.keys()),
                       help="Engine to test")
    parser.add_argument("--model", default="gpt-oss-20b", help="Model name")
    parser.add_argument("--prompt", default="What is artificial intelligence?", help="Test prompt")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test configuration
    config = {
        "tensor_parallel_size": 1,
        "tp_size": 1,
        "world_size": 1,
        "max_model_len": 2048
    }
    
    try:
        # Create adapter
        adapter = EngineFactory.create_adapter(args.engine, args.model, config)
        
        if not adapter.is_available():
            print(f"❌ {args.engine} is not available (not installed)")
            return 1
        
        print(f"🔧 Testing {args.engine} adapter...")
        
        # Initialize
        if not adapter.initialize():
            print(f"❌ Failed to initialize {args.engine}")
            return 1
        
        # Generate
        sampling_params = {
            "temperature": 0.8,
            "top_p": 0.95,
            "max_tokens": 100
        }
        
        results = adapter.generate([args.prompt], sampling_params)
        
        # Print results
        for result in results:
            print(f"\n✅ Generation successful!")
            print(f"Prompt: {result['prompt']}")
            print(f"Generated: {result['generated_text']}")
            print(f"Tokens: {result['tokens_generated']}")
            print(f"Speed: {result['tokens_per_second']:.1f} tokens/sec")
        
        # Cleanup
        adapter.cleanup()
        
        return 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())