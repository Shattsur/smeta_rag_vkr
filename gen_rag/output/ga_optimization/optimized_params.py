# Оптимальные параметры SOTA RAG
OPTIMIZED_PARAMS = {
    'bm25_k': 431, 'vec_k': 120, 'rrf_k': 160,
    'rerank_k': 20, 'rerank_threshold': 0.0139, 'final_k': 2,
    'reranker_model': "Shattsur/nemotron-smeta-4bit-adapter",
    'reranker_type': "auto",
    'fusion_strategy': "rrf",
    'hybrid_alpha': 0.44708131939037843,
    'retrieval_mode': "adaptive",
    'adaptive_threshold': 0.15141192934774367,
}
METRICS = {'cr': 0.8429, 'faith': 0.9611, 'ar': 0.8886}
SCORE = 2.9736 (Baseline: 2.8426)
