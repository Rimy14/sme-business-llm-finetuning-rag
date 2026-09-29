# 🚀 SME Daily Business LLM Fine-Tuning & RAG Framework

**Engineer / Assignee:** Rimaz Nowfel  
**Domain:** SME Daily Business (`SME-Daily-Business`)  
**Target Models:** Qwen 2.5-7B Instruct (`Qwen/Qwen2.5-7B-Instruct`) & Llama 3 8B Instruct (`meta-llama/Meta-Llama-3-8B-Instruct`)  
**Target Hardware:** Google Colab Tesla T4 GPU (15GB VRAM)

---

## 📅 12-Day Production Roadmap & Jira Task Mapping

| Day | Jira Key | Task Name | Status | Deliverables |
| :---: | :---: | :--- | :---: | :--- |
| **Day 1** | `KAN-13` | **Environment Setup + Data Collection** | 🟢 Completed | GPU verification, Drive mount, Library installs, Raw dataset fetch (`sme_raw_dataset.jsonl`) |
| **Day 2** | `KAN-17` | **Data Cleaning + QLoRA Configuration** | 🟢 Completed | Instruction-Response JSON formatting, 80/10/10 split, Qwen & Llama 4-bit config test |
| **Day 3** | `KAN-21` | **Writing Training Pipelines** | 🟢 Completed | SFTTrainer scripts (`train_qwen.py`, `train_llama.py`), T4 float32 adapter stability guard, W&B |
| **Day 4** | `KAN-26` | **Running v1 Training** | 🟢 Completed | Training v1 on 2,000 samples, VRAM monitoring, Checkpoint saves (`qwen_sme_v1`, `llama_sme_v1`) |
| **Day 5** | `KAN-30` / `KAN-31` | **Testing v1 + Inference Evaluation** | 🟢 Completed | BLEU/ROUGE/BERTScore eval, comparative matrix, adapter verification |
| **Day 6** | `KAN-35` | **Running v2 Training + Self-Testing v2** | 🟢 Completed | 550+ synthetic Q&A (`train_v2.json`), train v2 models, 30 domain tests, v1-to-v2 lift analysis |
| **Day 7** | `KAN-39` | **Preparing v3 Data (RAG-Aware Data)** | 🟢 Completed | Context injection into instruction pairs -> `train_v3.json`, enterprise SOPs in `rag_docs/` |
| **Day 8** | `KAN-43` | **Running v3 Training + Self-Testing v3** | 🟢 Completed | Train v3 models (`qwen_sme_v3`, `llama_sme_v3`), 30 domain tests, v2 vs v3 lift analysis |
| **Day 9** | `KAN-48` | **RAG Integration + Testing** | 🟢 Completed | ChromaDB vector store + `all-MiniLM-L6-v2` embeddings, 200 query evaluation (84.0% hit rate) |
| **Day 10** | `KAN-53` | **Running v4 Training + Testing** | 🟢 Completed | Train v4 RAG-aware production models (2.4k pairs), 100-query dual-model benchmark |
| **Day 11** | `KAN-57` | **Final Validation Across All Versions** | 🟢 Completed | Full collection evolution matrix (v1-v4 & RAG), progression curves & lift charts |
| **Day 12** | `KAN-61` | **Final Save + Master Report** | 🟢 Completed | 100-question validation, manual check audit, master registry (`master_model_registry.json`), executive summary report |

---

## 🏆 Final Benchmark & Head-to-Head Sprint Results

| Model Architecture / Stage | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU-4 | BERTScore | Hallucination Rate | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zero-Shot Base (Llama 3 8B)** | 48.10% | 27.50% | 37.80% | 23.10 | 85.20% | 31.0% | 1950ms |
| **Zero-Shot Base (Qwen 2.5 7B)** | 46.80% | 26.10% | 36.40% | 21.30 | 84.10% | 34.0% | 1820ms |
| **Qwen-SME v4 + ChromaDB RAG** | 76.50% | 59.10% | 66.80% | 48.40 | 93.40% | 1.5% | **2680ms** ⚡ |
| **Llama-SME v4 + ChromaDB RAG (Champion 🏆)** | **78.20%** | **62.40%** | **70.80%** | **51.60** | **94.10%** | **0.8%** 🛡️ | 2850ms |
| **Total Sprint Net Gain (Llama Base → v4+RAG)** | **+30.10%** | **+34.90%** | **+33.00%** | **+28.50 pts** | **+8.90%** | **-30.2% drop** | Production Ready |

---

## ⚡ Tesla T4 Hardware & Stability Guards

To prevent memory leaks and gradient scaler crashes on Google Colab T4:
1. **Force Cast bfloat16 to float32/float16:** T4 does not natively support bfloat16 compute. All buffers and non-quantized weights are cast to `torch.float32` / `torch.float16`.
2. **PEFT Adapter Initialization:** `model.config.torch_dtype = torch.float32` ensures LoRA adapters initialize cleanly in float32.
3. **QLoRA Parameters:**
   - 4-bit NormalFloat (`nf4`), Double Quantization (`bnb_4bit_use_double_quant=True`), Compute Dtype `torch.float16`.
   - LoRA Rank `r=16`, Alpha `lora_alpha=32`, Dropout `0.05` applied to all linear layers (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`).
4. **Batch Budget:**
   - `per_device_train_batch_size = 4`
   - `gradient_accumulation_steps = 4` (Effective batch size = 16)
   - Max train samples = 2,000 | Max validation samples = 200 (~1 hour per epoch on T4).

---

## 📁 Repository Directory Layout

```
AI SME/
├── configs/               # YAML configurations for Qwen and Llama QLoRA
├── data/
│   ├── raw/               # Raw downloaded datasets
│   ├── processed/         # 80/10/10 split JSONL data
│   ├── synthetic/         # Generated synthetic domain Q&A pairs
│   └── rag_docs/          # Enterprise SOPs and policy knowledge base
├── notebooks/             # Step-by-step Colab notebooks (Day 01 to Day 12)
├── src/
│   ├── data/              # Data collection, cleaning, and augmentation
│   ├── training/          # Training pipelines & T4 anti-crash utilities
│   ├── rag/               # ChromaDB indexer and retriever
│   └── evaluation/        # BLEU, ROUGE, and 100-query benchmarks
├── requirements.txt       # Production dependencies
└── README.md              # Project documentation
```
