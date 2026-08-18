# Reflection Cá Nhân — Lab 18: Production RAG Pipeline

**Họ và tên:** Đoàn Duy Chiến
**Mã sinh viên:** 2A202601366
**Ngày:** 18/08/2026

---

## Phần 1: Mapping Bài Giảng Vào Source Code

| Lecture Concept | Module | Hàm cụ thể | Observation & Insight |
|----------------|--------|-------------|-----------------------|
| Hierarchical Chunking | M1 | `chunk_hierarchical()` | Parent (2048) giữ ngữ cảnh rộng, Child (256) cho retrieval chính xác. Trên corpus thật (26 docs): 26 parents → 112 children, avg 202 chars/child. So với `chunk_basic` (51 chunks, avg 410) thì child chunk nhỏ hơn ~2x → embedding ít bị "loãng" chủ đề. |
| Semantic Chunking | M1 | `chunk_semantic()` | Cắt tại chỗ cosine similarity giữa 2 câu liền kề < 0.85. Kết quả 208 chunks, avg 99 chars, min 6 → threshold 0.85 quá "nhạy" với văn bản hành chính tiếng Việt (nhiều câu ngắn dạng gạch đầu dòng). Đây là lý do pipeline chọn hierarchical thay vì semantic. |
| Vietnamese Word Segmentation & Hybrid Fusion | M2 | `segment_vietnamese()`, `reciprocal_rank_fusion()` | `underthesea` tách từ ghép ("nghỉ phép" → `nghỉ_phép`) giúp BM25 hiểu ranh giới từ tiếng Việt. RRF `score = Σ 1/(k + rank)` với k=60 trộn 2 ranked list mà không cần normalize score (BM25 score không bounded, cosine ∈ [-1,1] → không thể cộng trực tiếp). |
| Cross-Encoder Reranking | M3 | `CrossEncoderReranker.rerank()` | `bge-reranker-v2-m3` encode chung cặp (query, doc) nên bắt được quan hệ mà bi-encoder bỏ sót. Chi phí: O(top_k) forward pass → chỉ khả thi vì đã lọc còn top-20 ở M2. Đây là lý do kiến trúc phải là *retrieve nhiều → rerank ít*, không phải rerank cả corpus. |
| RAGAS 4 Metrics Evaluation | M4 | `evaluate_ragas()`, `failure_analysis()` | 4 metric tách được lỗi retrieval (context_precision/recall) khỏi lỗi generation (faithfulness/answer_relevancy). Diagnostic Tree map metric tệ nhất → nguyên nhân gốc → fix cụ thể, thay vì chỉ nhìn 1 con số tổng. |
| Contextual Enrichment | M5 | `_enrich_single_call()`, `contextual_prepend()` | Prepend 1 câu mô tả vị trí chunk trong tài liệu (Anthropic contextual retrieval, giảm ~49% retrieval failure). Tối ưu chi phí: gộp summary + HyQA + context + metadata vào **1 API call** thay vì 4 → giảm 4x số request cho 112 chunks. |

---

## Phần 2: Khó Khăn Gặp Phải & Cách Giải Quyết

1. **`underthesea` nối từ ghép bằng dấu `_` làm BM25 không match**
   - *Lỗi:* `word_tokenize(format="text")` trả về `"nghỉ_phép"`. BM25 tokenize bằng `.split()` → corpus có token `nghỉ_phép`, còn query người dùng ("nghỉ phép") tách thành 2 token `nghỉ`, `phép` → **không khớp token nào**, BM25 score = 0.
   - *Fix:* thêm `.replace("_", " ")` ngay sau segmentation trong `segment_vietnamese()`, áp dụng cho cả index và query để tokenization đối xứng.

2. **Reranker crash với `transformers>=5.0`**
   - *Lỗi:* `FlagEmbedding.FlagReranker` gọi `XLMRobertaTokenizer` theo API cũ → lỗi khi khởi tạo.
   - *Fix:* dùng `sentence_transformers.CrossEncoder("BAAI/bge-reranker-v2-m3")` (`model.predict(pairs)`), API ổn định với transformers 5.x.

3. **OpenRouter không có endpoint `/embeddings` → RAGAS `answer_relevancy` chết**
   - *Lỗi:* RAGAS mặc định gọi OpenAI embeddings và dùng model id trần `gpt-4o-mini`; OpenRouter yêu cầu prefix `openai/` và không hỗ trợ embeddings.
   - *Fix:* truyền tường minh vào `evaluate()`: `llm=LangchainLLMWrapper(ChatOpenAI(base_url=OPENROUTER_BASE_URL, model=LLM_MODEL))` và `embeddings=LangchainEmbeddingsWrapper(_SharedEncoderEmbeddings(...))` — một wrapper duck-typed chạy `bge-m3` **local** thay cho OpenAI embeddings.

4. **RAM: `bge-m3` (~2.3GB) bị load 2 lần (M2 dense search + M4 answer_relevancy)**
   - *Lỗi:* máy chỉ còn ~3.4GB RAM trống → nguy cơ OOM giữa lúc evaluate.
   - *Fix:* thêm `config.get_encoder()` với `_ENCODER_CACHE` ở module level, cả `DenseSearch` và `_SharedEncoderEmbeddings` cùng lấy 1 instance.

5. **Một câu hỏi lỗi làm chết cả run evaluation**
   - *Fix:* `evaluate(..., raise_exceptions=False)` để metric lỗi trả `NaN`, cộng thêm hàm `_clean()` map `NaN/None → 0.0` — điểm bị trừ đúng chỗ nhưng report vẫn hoàn tất.

6. **`UnicodeEncodeError` (cp1252) khi in tiếng Việt/emoji trên Windows**
   - *Fix:* chạy với `PYTHONIOENCODING=utf-8`.

7. **`qdrant-client >= 1.10` bỏ `.search()`**
   - *Fix:* dùng `client.query_points(collection_name=..., query=vector, limit=top_k)` và đọc `response.points`.

---

## Phần 3: Action Plan Cho Project Cá Nhân

### Project: Hệ thống RAG hỏi đáp văn bản pháp luật & quy định nội bộ (tiếng Việt)

| Tuần | Việc cần làm | Kỹ thuật từ Lab 18 |
|------|--------------|--------------------|
| 1 | Ingest + OCR tài liệu (2/28 PDF trong lab là scan ảnh, không có text layer → bắt buộc OCR trước khi chunk) | `load_documents()` + thêm Tesseract/PaddleOCR |
| 2 | Chunking: Structure-Aware theo Điều/Khoản, kết hợp Hierarchical parent-child | `chunk_structure_aware()` + `chunk_hierarchical()` |
| 3 | Enrichment: contextual prepend + auto metadata (số hiệu văn bản, ngày ban hành, phiên bản) trong 1 LLM call | `_enrich_single_call()` |
| 4 | Hybrid Search: BM25 (segmented) + Qdrant dense, hợp nhất bằng RRF | `HybridSearch`, `reciprocal_rank_fusion()` |
| 5 | Rerank top-20 → top-3~5 bằng `bge-reranker-v2-m3`, thêm metadata filter theo phiên bản văn bản để tránh trộn v1/v2 | `CrossEncoderReranker` |
| 6 | Dựng test set ≥30 câu + RAGAS 4 metrics chạy tự động trên CI, đặt ngưỡng chặn merge (VD faithfulness < 0.8 → fail) | `evaluate_ragas()`, `failure_analysis()` |

### Nguyên tắc rút ra để mang sang project thật
- **Đo trước khi tối ưu:** luôn có baseline naive (fixed chunk + dense-only) để mỗi kỹ thuật thêm vào phải chứng minh được delta trên RAGAS, không thêm theo cảm tính.
- **Retrieve rộng, rerank hẹp:** top-20 hybrid → top-3 cross-encoder là điểm cân bằng giữa recall và precision/latency.
- **Metric chỉ đúng nguyên nhân, không chỉ đúng/sai:** faithfulness thấp → sửa prompt/temperature; context_recall thấp → sửa chunking/BM25; context_precision thấp → sửa rerank/filter.
- **Fallback ở mọi lớp gọi LLM/model ngoài:** mọi hàm enrichment đều có nhánh extractive không cần API, để pipeline không sập khi hết quota.

---

## Tự Đánh Giá

| Tiêu chí | Điểm tự đánh giá | Ghi chú |
|----------|------------------|---------|
| Hoàn thành code (M1–M5) | 10/10 | Không còn TODO; toàn bộ test tự động pass |
| Hiểu bản chất kỹ thuật | 9/10 | Nắm rõ RRF, cross-encoder vs bi-encoder; cần đọc thêm về ColBERT / multi-vector của bge-m3 |
| Chất lượng phân tích lỗi | 9/10 | Failure analysis dựa trên số liệu RAGAS thật, không phỏng đoán |
| Đóng góp / tự chủ | 9/10 | Tự debug các lỗi môi trường (OpenRouter embeddings, transformers 5.x, RAM) |
