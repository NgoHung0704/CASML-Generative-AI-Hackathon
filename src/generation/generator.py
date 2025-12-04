"""
Generation module - LLM integration and answer generation
"""

import tensorflow as tf
from typing import List, Dict
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM, TFAutoModelForCausalLM
import numpy as np


class LLMGenerator:
    """
    Wrapper for LLM-based answer generation using TensorFlow.
    """
    
    def __init__(self, config):
        """
        Initialize LLM generator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.model_name = config.get('generation.model.model_name')
        self.device = config.get('generation.model.device', 'CPU')
        self.use_mixed_precision = config.get('generation.model.use_mixed_precision', False)
        
        # Configure TensorFlow
        if 'GPU' in self.device:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    # Enable memory growth
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    
                    # Enable mixed precision for faster inference
                    if self.use_mixed_precision:
                        tf.keras.mixed_precision.set_global_policy('mixed_float16')
                        
                except RuntimeError as e:
                    print(f"GPU configuration error: {e}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Determine model type (seq2seq vs causal)
        if 'flan' in self.model_name.lower() or 't5' in self.model_name.lower():
            model_class = TFAutoModelForSeq2SeqLM
        else:
            model_class = TFAutoModelForCausalLM
        
        # Load TensorFlow model
        self.model = model_class.from_pretrained(self.model_name)
        
        # Generation parameters
        self.max_new_tokens = config.get('generation.inference.max_new_tokens', 256)
        self.temperature = config.get('generation.inference.temperature', 0.3)
        self.top_p = config.get('generation.inference.top_p', 0.9)
        self.top_k = config.get('generation.inference.top_k', 50)
        self.do_sample = config.get('generation.inference.do_sample', True)
        self.num_beams = config.get('generation.inference.num_beams', 1)
        self.repetition_penalty = config.get('generation.inference.repetition_penalty', 1.2)
    
    def generate(self, prompt: str) -> str:
        """
        Generate answer from prompt.
        
        Args:
            prompt: Input prompt
        
        Returns:
            Generated answer
        """
        inputs = self.tokenizer(prompt, return_tensors='tf', truncation=True, max_length=2048)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            do_sample=self.do_sample,
            num_beams=self.num_beams,
            repetition_penalty=self.repetition_penalty
        )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # For encoder-decoder models, the output already excludes the input
        # For causal models, we need to remove the prompt
        if 'flan' not in self.model_name.lower() and 't5' not in self.model_name.lower():
            # Remove the prompt from the answer for causal models
            answer = answer[len(prompt):].strip()
        
        return answer


class RAGPipeline:
    """
    End-to-end RAG pipeline combining retrieval and generation.
    """
    
    def __init__(self, config, retriever, generator):
        """
        Initialize RAG pipeline.
        
        Args:
            config: Configuration object
            retriever: Retriever instance
            generator: LLMGenerator instance
        """
        self.config = config
        self.retriever = retriever
        self.generator = generator
        
        self.prompt_template = config.get('generation.prompt.template')
        self.max_context_length = config.get('generation.prompt.max_context_length', 2048)
    
    def build_prompt(self, query: str, contexts: List[str]) -> str:
        """
        Build prompt from query and retrieved contexts.
        
        Args:
            query: User query
            contexts: Retrieved context chunks
        
        Returns:
            Formatted prompt
        """
        # Combine contexts
        combined_context = '\n\n'.join(contexts)
        
        # Truncate if too long
        if len(combined_context) > self.max_context_length:
            combined_context = combined_context[:self.max_context_length]
        
        # Format prompt
        prompt = self.prompt_template.format(
            context=combined_context,
            question=query
        )
        
        return prompt
    
    def answer_question(self, query: str) -> Dict[str, any]:
        """
        Answer a question using RAG.
        
        Args:
            query: User question
        
        Returns:
            Dictionary with answer and metadata
        """
        # Step 1: Retrieve relevant contexts
        retrieval_results = self.retriever.retrieve(query)
        contexts = [r['chunk'] for r in retrieval_results]
        
        # Step 2: Build prompt
        prompt = self.build_prompt(query, contexts)
        
        # Step 3: Generate answer
        answer = self.generator.generate(prompt)
        
        return {
            'query': query,
            'answer': answer,
            'contexts': contexts,
            'retrieval_results': retrieval_results,
            'prompt': prompt
        }
