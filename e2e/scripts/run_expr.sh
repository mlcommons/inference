python -u single_shot_retrieval.py --retrieval_method vector --db vector_html_hnsw_len2048_ov32_word --eval --no-rerank --retrieval_strategy relative --relative_ratio 0.511 --generate-answer |& tee llm_vector_hnsw_len2048_no_rerank.txt
mv result_single_shot.json result_single_shot_llm_vector_hnsw_len2048_word_no_rerank_rel0.511.json
python -u single_shot_retrieval.py --retrieval_method vector --db vector_html_hnsw_len4096_ov32_word --eval --no-rerank --retrieval_strategy relative --relative_ratio 0.511 --generate-answer |& tee llm_vector_hnsw_len4096_no_rerank.txt
mv result_single_shot.json result_single_shot_llm_vector_hnsw_len4096_word_no_rerank_rel0.511.json
python -u evaluate.py result_single_shot_llm_vector_hnsw_len256_word.json |& tee score_single_shot_llm_vector_hnsw_len256_word.txt
python -u evaluate.py result_single_shot_llm_vector_hnsw_len2048_word_no_rerank_rel0.511.json |& tee score_single_shot_llm_vector_hnsw_len2048_word.txt
python -u evaluate.py result_single_shot_llm_vector_hnsw_len4096_word_no_rerank_rel0.511.json |& tee score_single_shot_llm_vector_hnsw_len4096_word.txt
