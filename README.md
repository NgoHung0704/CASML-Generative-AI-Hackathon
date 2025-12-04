# CASML - Generative AI Hackathon RAG System

Hệ thống **Retrieval-Augmented Generation (RAG)** modular cho cuộc thi [CASML Generative AI Hackathon](https://www.kaggle.com/competitions/casml-generative-ai-hackathon) trên Kaggle.

## 📋 Mục lục

- [Giới thiệu](#giới-thiệu)
- [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Cài đặt](#cài-đặt)
- [Sử dụng](#sử-dụng)
- [Cấu hình](#cấu-hình)
- [Mô-đun chi tiết](#mô-đun-chi-tiết)
- [Tùy chỉnh & mở rộng](#tùy-chỉnh--mở-rộng)

---

## 🎯 Giới thiệu

Dự án này xây dựng một hệ thống hỏi–đáp (Q&A) sử dụng **RAG Pipeline** để:
1. **Truy xuất** các đoạn văn liên quan từ corpus (sách/tài liệu)
2. **Sinh câu trả lời** chính xác dựa trên ngữ cảnh được truy xuất bằng LLM

### Đặc điểm chính:
- ✅ **Modular**: Dễ dàng thay đổi embedding model, retriever, hoặc LLM
- ✅ **Reproducible**: Seed cố định, cấu hình rõ ràng
- ✅ **Flexible**: Hỗ trợ nhiều chiến lược retrieval (dense, sparse, hybrid)
- ✅ **Optimized**: Thiết kế cho GPU hạn chế, hỗ trợ quantization
- ✅ **Kaggle-ready**: Script tự động tạo submission file

---

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────────┐
│                         RAG PIPELINE                             │
└─────────────────────────────────────────────────────────────────┘
                                ↓
    ┌──────────────────────────────────────────────────────┐
    │  1. DATA INGESTION LAYER                             │
    │     - Load corpus (books/documents)                  │
    │     - Parse & clean text                             │
    │     - Load Q&A dataset                               │
    └──────────────────┬───────────────────────────────────┘
                       ↓
    ┌──────────────────────────────────────────────────────┐
    │  2. INDEXING LAYER                                   │
    │     - Text chunking (semantic/fixed-size)            │
    │     - Embedding generation (sentence-transformers)   │
    │     - Index building (FAISS/BM25/Hybrid)             │
    └──────────────────┬───────────────────────────────────┘
                       ↓
    ┌──────────────────────────────────────────────────────┐
    │  3. RETRIEVAL LAYER                                  │
    │     - Query embedding                                │
    │     - Top-K retrieval (dense/sparse/hybrid)          │
    │     - Re-ranking (optional: cross-encoder)           │
    └──────────────────┬───────────────────────────────────┘
                       ↓
    ┌──────────────────────────────────────────────────────┐
    │  4. GENERATION LAYER                                 │
    │     - Context assembly (retrieved chunks)            │
    │     - Prompt engineering                             │
    │     - LLM inference (HuggingFace/OpenAI/local)       │
    └──────────────────┬───────────────────────────────────┘
                       ↓
    ┌──────────────────────────────────────────────────────┐
    │  5. EVALUATION & SUBMISSION LAYER                    │
    │     - Metric calculation (BLEU/ROUGE/Similarity)     │
    │     - Generate submission CSV                        │
    └──────────────────────────────────────────────────────┘
```

---

## 📁 Cấu trúc dự án

```
CASML-Generative-AI-Hackathon/
│
├── data/                          # Dữ liệu
│   ├── raw/                       # Corpus & Q&A gốc từ Kaggle
│   ├── processed/                 # Chunks đã xử lý
│   ├── embeddings/                # Vector embeddings
│   ├── indexes/                   # FAISS/BM25 indexes
│   └── submissions/               # Submission files cho Kaggle
│
├── src/                           # Source code chính
│   ├── config/                    # Quản lý cấu hình
│   │   ├── __init__.py
│   │   └── config_manager.py      # Load config.yaml & .env
│   │
│   ├── ingestion/                 # Load & preprocess dữ liệu
│   │   ├── __init__.py
│   │   └── data_loader.py         # CorpusLoader, QADataLoader
│   │
│   ├── indexing/                  # Chunking & embedding
│   │   ├── __init__.py
│   │   └── indexer.py             # TextChunker, EmbeddingGenerator, IndexBuilder
│   │
│   ├── retrieval/                 # Truy xuất documents
│   │   ├── __init__.py
│   │   └── retriever.py           # Dense/Sparse/Hybrid retrieval + Re-ranking
│   │
│   ├── generation/                # LLM inference
│   │   ├── __init__.py
│   │   └── generator.py           # LLMGenerator, RAGPipeline
│   │
│   ├── evaluation/                # Metrics & submission
│   │   ├── __init__.py
│   │   └── evaluator.py           # Evaluator, SubmissionGenerator
│   │
│   ├── utils/                     # Helper functions
│   │   ├── __init__.py
│   │   └── helpers.py             # Logging, seeding, timing
│   │
│   └── __init__.py
│
├── scripts/                       # Executable scripts
│   ├── download_data.py           # Download Kaggle data
│   ├── build_index.py             # Build embeddings & indexes
│   ├── evaluate.py                # Evaluate on training set
│   ├── generate_predictions.py    # Generate test predictions
│   └── run_pipeline.py            # Run end-to-end pipeline
│
├── notebooks/                     # Jupyter notebooks
│   └── (EDA, experiments, visualization)
│
├── tests/                         # Unit tests
│   └── (pytest tests)
│
├── config.yaml                    # Cấu hình chính
├── .env.example                   # Template cho biến môi trường
├── requirements.txt               # Python dependencies
├── .gitignore
└── README.md                      # File này
```

---

## 🚀 Cài đặt

### 1. Clone repository
```bash
git clone https://github.com/your-username/CASML-Generative-AI-Hackathon.git
cd CASML-Generative-AI-Hackathon
```

### 2. Tạo virtual environment
```powershell
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

**Lưu ý**: 
- Nếu có GPU CUDA, dùng `tensorflow-gpu` và `faiss-gpu`:
  ```bash
  pip install tensorflow-gpu==2.15.0
  pip install faiss-gpu
  ```
- Kiểm tra CUDA và cuDNN tương thích: https://www.tensorflow.org/install/source#gpu

### 4. Cấu hình API keys (tùy chọn)
```bash
cp .env.example .env
# Chỉnh sửa .env và thêm API keys (nếu dùng OpenAI/Anthropic/Kaggle)
```

---

## 💻 Sử dụng

### Pipeline hoàn chỉnh (Recommended)

Chạy toàn bộ pipeline từ data đến submission:

```bash
python scripts/run_pipeline.py
```

### Từng bước riêng lẻ

#### Bước 1: Download dữ liệu từ Kaggle
```bash
python scripts/download_data.py
```
*Yêu cầu: KAGGLE_USERNAME và KAGGLE_KEY trong .env*

#### Bước 2: Build search index
```bash
python scripts/build_index.py
```
Tạo chunks, embeddings, và FAISS/BM25 indexes.

#### Bước 3: Evaluate trên training set
```bash
python scripts/evaluate.py
```
Đánh giá model với BLEU, ROUGE, BERTScore trên tập train.

#### Bước 4: Generate predictions cho test set
```bash
python scripts/generate_predictions.py
```
Tạo file submission CSV trong `data/submissions/`.

---

## ⚙️ Cấu hình

Tất cả cấu hình được quản lý trong **`config.yaml`**.

### Các phần chính:

#### 1. **Indexing**
```yaml
indexing:
  chunking:
    strategy: "semantic"  # fixed, semantic, sentence, paragraph
    chunk_size: 512
    chunk_overlap: 50
  
  embedding:
    model_name: "sentence-transformers/all-MiniLM-L6-v2"
    # Hoặc dùng fine-tuned model:
    # model_name: "your-username/your-finetuned-model"
    device: "cuda"
    batch_size: 32
  
  index:
    type: "hybrid"  # faiss, bm25, hybrid
```

#### 2. **Retrieval**
```yaml
retrieval:
  strategy: "hybrid"  # dense, sparse, hybrid
  top_k: 5
  dense_weight: 0.6
  sparse_weight: 0.4
  
  use_reranker: true
  reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  rerank_top_k: 3
```

#### 3. **Generation**
```yaml
generation:
  model:
    provider: "huggingface"
    model_name: "google/flan-t5-base"
    # Hoặc: "meta-llama/Llama-2-7b-chat-hf"
    # Hoặc: "your-username/your-finetuned-llm"
    device: "cuda"
    load_in_8bit: false  # Quantization để tiết kiệm VRAM
  
  inference:
    max_new_tokens: 256
    temperature: 0.3
    top_p: 0.9
  
  prompt:
    template: |
      Answer the question based on the context below.
      
      Context: {context}
      Question: {question}
      Answer:
```

#### 4. **Evaluation**
```yaml
evaluation:
  metrics:
    - "bleu"
    - "rouge"
    - "bertscore"
```

---

## 🧩 Mô-đun chi tiết

### 1. **Config Module** (`src/config/`)
- **Vai trò**: Load cấu hình từ `config.yaml` và `.env`
- **Sử dụng**:
  ```python
  from src.config import get_config
  config = get_config()
  embedding_model = config.get('indexing.embedding.model_name')
  ```

### 2. **Ingestion Module** (`src/ingestion/`)
- **Vai trò**: Load corpus và Q&A dataset, preprocessing text
- **Classes**:
  - `CorpusLoader`: Load và clean corpus
  - `QADataLoader`: Load train/test Q&A data
- **Flow**: Raw data → Cleaned text

### 3. **Indexing Module** (`src/indexing/`)
- **Vai trò**: Chunk text, tạo embeddings, build search indexes
- **Classes**:
  - `TextChunker`: Chia corpus thành chunks (fixed/semantic/sentence/paragraph)
  - `EmbeddingGenerator`: Tạo embeddings với sentence-transformers
  - `IndexBuilder`: Build FAISS (dense), BM25 (sparse), hoặc hybrid index
- **Flow**: Text → Chunks → Embeddings → Index

### 4. **Retrieval Module** (`src/retrieval/`)
- **Vai trò**: Truy xuất top-K documents liên quan cho query
- **Class**: `Retriever`
  - Dense retrieval: FAISS similarity search
  - Sparse retrieval: BM25 keyword matching
  - Hybrid: Kết hợp dense + sparse với weighted fusion
  - Re-ranking: Cross-encoder để cải thiện ranking
- **Flow**: Query → Embeddings → Search → Re-rank → Top-K chunks

### 5. **Generation Module** (`src/generation/`)
- **Vai trò**: Sinh câu trả lời từ LLM
- **Classes**:
  - `LLMGenerator`: Wrapper cho HuggingFace models (T5, Llama, GPT, v.v.)
  - `RAGPipeline`: Kết hợp retrieval + generation
- **Flow**: Query → Retrieve contexts → Build prompt → LLM inference → Answer

### 6. **Evaluation Module** (`src/evaluation/`)
- **Vai trò**: Đánh giá predictions và tạo submission file
- **Classes**:
  - `Evaluator`: Tính BLEU, ROUGE, BERTScore, Exact Match
  - `SubmissionGenerator`: Tạo CSV submission cho Kaggle
- **Flow**: Predictions + References → Metrics / Submission CSV

### 7. **Utils Module** (`src/utils/`)
- **Vai trò**: Helper functions
- **Functions**:
  - `setup_logging()`: Cấu hình logging
  - `set_seed()`: Set random seed cho reproducibility
  - `get_device()`: Auto-detect CUDA/MPS/CPU
  - `save_results()` / `load_results()`: Save/load JSON results

---

## 🛠️ Tùy chỉnh & mở rộng

### 1. Sử dụng fine-tuned embedding model từ HuggingFace

Chỉnh sửa `config.yaml`:
```yaml
indexing:
  embedding:
    model_name: "your-username/your-finetuned-embedding-model"
    backend: "tensorflow"
```

### 2. Sử dụng fine-tuned LLM

Chỉnh sửa `config.yaml`:
```yaml
generation:
  model:
    model_name: "your-username/your-finetuned-llm"
    backend: "tensorflow"
    use_mixed_precision: true  # Nếu GPU nhỏ
```

### 3. Thay đổi chunking strategy

```yaml
indexing:
  chunking:
    strategy: "sentence"  # Thử sentence-based chunking
```

### 4. Điều chỉnh retrieval strategy

```yaml
retrieval:
  strategy: "dense"  # Chỉ dùng dense retrieval
  top_k: 10          # Tăng số chunks retrieve
```

### 5. Custom prompt template

Chỉnh sửa trong `config.yaml`:
```yaml
generation:
  prompt:
    template: |
      You are a helpful assistant. Answer concisely.
      
      Context:
      {context}
      
      Question: {question}
      
      Answer:
```

### 6. Thêm metrics mới

Chỉnh sửa `src/evaluation/evaluator.py` và implement metric tùy chỉnh trong class `Evaluator`.

---

## 🧪 Testing

Chạy unit tests:
```bash
pytest tests/
```

---

## 📊 Experiment Tracking (Optional)

### Sử dụng Weights & Biases

1. Cấu hình trong `config.yaml`:
```yaml
logging:
  use_wandb: true
  wandb_project: "casml-rag-hackathon"
```

2. Set API key trong `.env`:
```
WANDB_API_KEY=your_wandb_key
```

3. Log metrics tự động khi chạy scripts.

---

## 🛠️ Troubleshooting

### GPU Out of Memory
- Giảm `batch_size` trong config
- Bật mixed precision: `use_mixed_precision: true` trong generation config
- Đặt `gpu_memory_limit` trong resources config (MB)
- Giảm `chunk_size` và `top_k`

### TensorFlow GPU không nhận
- Kiểm tra CUDA và cuDNN đã cài đúng phiên bản
- Chạy: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
- Bật `gpu_memory_growth: true` trong config

### FAISS Import Error
- Cài `faiss-gpu` nếu có CUDA:
  ```bash
  pip uninstall faiss-cpu
  pip install faiss-gpu
  ```

### Kaggle API Error
- Đảm bảo đã accept competition rules trên Kaggle
- Kiểm tra `KAGGLE_USERNAME` và `KAGGLE_KEY` trong `.env`

---

## 📚 Tài liệu tham khảo

- **TensorFlow**: https://www.tensorflow.org/guide
- **HuggingFace Transformers**: https://huggingface.co/docs/transformers
- **Sentence Transformers**: https://www.sbert.net/
- **FAISS**: https://github.com/facebookresearch/faiss
- **LangChain**: https://python.langchain.com/
- **RAG Papers**: [RAG (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)

---

## 📄 License

MIT License

---

## 👥 Contributors

CASML Team - INSA Lyon

---

## 🙏 Acknowledgments

Cuộc thi được tổ chức bởi CASML trên nền tảng Kaggle.

