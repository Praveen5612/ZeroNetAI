import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from typing import Dict, Any, Optional
import gc
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.device = self._setup_device()
        self.cache_dir = Path("c:/Documents/projects/AI/models/cache")
        self.max_memory = self._get_available_memory()
        
    def _setup_device(self) -> torch.device:
        """Setup optimal device with memory management"""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            return torch.device("cuda")
        return torch.device("cpu")
    
    def _get_available_memory(self) -> int:
        """Get available system memory"""
        if self.device.type == "cuda":
            return torch.cuda.get_device_properties(0).total_memory
        try:
            import psutil
            return psutil.virtual_memory().total
        except ImportError:
            return 8589934592  # Default to 8GB if psutil not available
    
    def _optimize_model(self):
        """Apply model-specific optimizations"""
        if self.device.type == "cuda":
            self.model.half()  # FP16
            torch.cuda.empty_cache()
        
        try:
            # Enable gradient checkpointing
            self.model.gradient_checkpointing_enable()
        except Exception as e:
            logger.warning(f"Gradient checkpointing not available: {e}")
        
        try:
            # Enable model parallelism if available
            if hasattr(self.model, "parallelize"):
                self.model.parallelize()
        except Exception as e:
            logger.warning(f"Model parallelization not available: {e}")

    def _get_optimization_status(self) -> Dict[str, bool]:
        """Get status of applied optimizations"""
        try:
            return {
                "fp16_enabled": self.model.dtype == torch.float16 if self.model else False,
                "cuda_optimized": self.device.type == "cuda",
                "gradient_checkpointing": getattr(self.model, "is_gradient_checkpointing", False),
                "parallel_enabled": getattr(self.model, "is_parallelized", False)
            }
        except Exception as e:
            logger.error(f"Error getting optimization status: {e}")
            return {
                "fp16_enabled": False,
                "cuda_optimized": False,
                "gradient_checkpointing": False,
                "parallel_enabled": False
            }
    
    def load_model(self):
        """Load model with advanced optimizations"""
        logger.info(f"Loading model {self.model_name} on {self.device}")
        try:
            # Clear CUDA cache
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            # Create cache directory
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Load tokenizer with caching
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                use_fast=True
            )
            
            # Load model with optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                device_map="auto",
                use_cache=True,
                quantization_config=self._get_quantization_config()
            )
            
            # Apply additional optimizations
            self._optimize_model()
            logger.info("Model loaded with advanced optimizations")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _get_quantization_config(self):
        """Get quantization configuration based on device"""
        if self.device.type == "cuda":
            from transformers import BitsAndBytesConfig
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
        return None
    
    def _format_prompt(self, prompt: str, context: Optional[str] = None) -> str:
        """Format prompt with context"""
        # Define conversation template
        conversation_template = """<|system|>You are a helpful AI assistant. Be friendly, concise, and natural in your responses.
<|user|>{prompt}
<|assistant|>"""
        
        if context:
            return conversation_template.format(prompt=f"{context}\n{prompt}")
        return conversation_template.format(prompt=prompt)

    def _get_generation_config(self) -> GenerationConfig:
        """Get optimized generation configuration"""
        return GenerationConfig(
            max_length=2048,
            min_length=10,
            do_sample=True,
            temperature=0.9,  # More creative responses
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            num_beams=1,  # Single beam for more natural responses
            length_penalty=1.0,
            early_stopping=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

    def _clean_response(self, response: str) -> str:
        """Clean and format the response"""
        # Remove system and user prompts
        response = response.replace("<|system|>", "").replace("<|user|>", "").replace("<|assistant|>", "")
        
        # Clean up any remaining artifacts
        response = response.split("User:")[0].split("Assistant:")[0].strip()
        
        # Handle greetings specially
        if any(greeting in response.lower() for greeting in ["hi", "hello", "hey"]):
            return "Hello! How may I help you today?"
            
        # Ensure response starts naturally
        if len(response) < 10:
            return "I'm here to help. What would you like to know?"
            
        return response.strip()

    def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        """Generate response with advanced parameters"""
        if not self.model or not self.tokenizer:
            return "Error: Model not loaded"
        
        try:
            # Handle common greetings
            lower_prompt = prompt.lower().strip()
            if lower_prompt in ["hi", "hello", "hey", "hi there", "hello there"]:
                return "Hello! How may I help you today?"
            
            # Regular response generation
            formatted_prompt = self._format_prompt(prompt, context)
            inputs = self._prepare_inputs(formatted_prompt)
            generation_config = self._get_generation_config()
            
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=self.device.type=="cuda"):
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config,
                    max_new_tokens=512,
                    num_return_sequences=1
                )
            
            response = self._process_response(outputs, formatted_prompt)
            
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error. How else can I assist you?"
    
    def _clean_response(self, response: str) -> str:
        """Clean and format the response"""
        # Remove any potential artifacts
        response = response.split("User:")[0].split("Assistant:")[0]
        return response.strip()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information"""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "dtype": str(self.model.dtype) if self.model else None,
            "parameters": sum(p.numel() for p in self.model.parameters()) if self.model else 0,
            "memory_usage": self._get_memory_usage(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "optimization_status": self._get_optimization_status()
        }
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        if self.device.type == "cuda":
            return {
                "allocated": torch.cuda.memory_allocated(0) / 1024**2,
                "cached": torch.cuda.memory_reserved(0) / 1024**2
            }
        return {"system_memory": self.max_memory / 1024**2}