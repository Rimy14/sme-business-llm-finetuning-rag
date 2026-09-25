"""
SME End-to-End RAG Pipeline Orchestrator (Day 9 - KAN-48)
Connects ChromaDB Vector Retrieval with fine-tuned LoRA models (Qwen-SME-v3 / Llama-SME-v3).
"""

import time
import torch
from typing import Dict, Any, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from .vector_store import SMEVectorStore


class SMERAGPipeline:
    """End-to-End RAG Orchestrator for SME Business Assistance."""

    def __init__(
        self,
        vector_store: SMEVectorStore,
        base_model_id: str,
        adapter_path: str,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
        load_in_4bit: bool = True
    ):
        self.vector_store = vector_store
        self.base_model_id = base_model_id
        self.adapter_path = adapter_path

        print(f"🔧 Initializing Tokenizer for: {adapter_path} (Fallback: {base_model_id})")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        print(f"🚀 Loading Base Model ({base_model_id}) with 4-bit QLoRA...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype
        ) if load_in_4bit else None

        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        print(f"⚡ Attaching Fine-Tuned v3 Adapter from: {adapter_path}...")
        self.model = PeftModel.from_pretrained(self.base_model, adapter_path)
        self.model.eval()
        print("✅ SME RAG Pipeline successfully loaded and ready for inference!")

    def build_prompt(self, query: str, context: Optional[str] = None) -> str:
        """Construct a structured chat prompt adhering to the SME Assistant persona."""
        sys_msg = (
            "You are an expert SME daily business assistant specializing in enterprise SOPs, "
            "financial controls, cash flow management, taxation, and supply chain operations. "
            "Provide precise, actionable, and policy-compliant guidance based on the provided context."
        )

        user_content = query
        if context and context.strip():
            user_content = f"Context Reference:\n{context}\n\nQuestion:\n{query}"

        return self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_content}
            ],
            tokenize=False,
            add_generation_prompt=True
        )

    @torch.no_grad()
    def generate(
        self,
        query: str,
        use_rag: bool = True,
        top_k: int = 3,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.9
    ) -> Dict[str, Any]:
        """
        Execute RAG generation for a query.
        Returns generated answer, retrieved source passages, similarity scores, and execution latency.
        """
        start_time = time.time()
        retrieved_sources = []
        context_str = ""

        # Step 1: Retrieval
        if use_rag and self.vector_store is not None:
            retrieved_sources = self.vector_store.retrieve(query, top_k=top_k)
            context_str = self.vector_store.format_retrieved_context(retrieved_sources)

        # Step 2: Prompt Formatting
        prompt = self.build_prompt(query, context=context_str if use_rag else None)

        # Step 3: Model Inference
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        in_len = inputs["input_ids"].shape[1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            temperature=temperature if temperature > 0.0 else None,
            top_p=top_p if temperature > 0.0 else None,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        decoded = self.tokenizer.decode(outputs[0][in_len:], skip_special_tokens=True).strip()
        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "query": query,
            "answer": decoded,
            "use_rag": use_rag,
            "retrieved_sources": retrieved_sources,
            "num_retrieved": len(retrieved_sources),
            "top_similarity": retrieved_sources[0]["similarity"] if retrieved_sources else 0.0,
            "latency_ms": elapsed_ms
        }
