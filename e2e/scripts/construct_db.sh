python -u single_shot_retrieval.py --ingest passages/doc_html_len512_overlap32_word.json --db  vector_hnsw_len512_ov32_word --retrieval_method vector --num_embedding_devices 8 
python -u single_shot_retrieval.py --ingest passages/doc_html_len1024_overlap32_word.json --db vector_hnsw_len1024_ov32_word --retrieval_method vector --num_embedding_devices 8
python -u single_shot_retrieval.py --ingest passages/doc_html_len2048_overlap32_word.json --db vector_hnsw_len2048_ov32_word --retrieval_method vector --num_embedding_devices 8
python -u single_shot_retrieval.py --ingest passages/doc_html_len4096_overlap32_word.json --db vector_hnsw_len4096_ov32_word --retrieval_method vector --num_embedding_devices 8
python -u single_shot_retrieval.py --ingest passages/doc_html_len8192_overlap32_word.json --db vector_hnsw_len8192_ov32_word --retrieval_method vector --num_embedding_devices 8
