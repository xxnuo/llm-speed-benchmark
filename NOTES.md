# Model Performance Summary

| Model | Time To First Token | Prompt Tok/s | Response Tok/s | Num Response Tokens | Avg Tokens per Chunk | Avg Time Between Chunks |
| --- | --- | --- | --- | --- | --- | --- |
| SGLang 引擎，默认模式，占用约 40G 显存，Qwen3-4B-Instruct-2507-Int4-W4A16 | 0.09 +/- 0.01 | 916.23 +/- 7.46 | 52.41 +/- 0.21 | 1270.00 +/- 62.75 | 4.41 +/- 0.10 | 0.02 +/- 0.00 |
| ollama，测试 qwen3:4b-instruct-2507-q4_K_M | 0.20 +/- 0.00 | 396.56 +/- 13.92 | 35.62 +/- 5.88 | 1448.00 +/- 129.75 | 4.42 +/- 0.15 | 0.03 +/- 0.00 |
| vllm aipod-trans,MAX_MODEL_LEN=4096，占用约 10G 显存，Qwen3-4B-Instruct-2507-Int4-W4A16 | 0.04 +/- 0.00 | 1984.32 +/- 46.04 | 46.26 +/- 0.04 | 1562.00 +/- 0.00 | 4.33 +/- 0.00 | 0.02 +/- 0.00 |
| vllm 新底包，enforce_eager | 0.06 +/- 0.00 | 1234.27 +/- 8.81 | 19.16 +/- 0.05 | 1325.50 +/- 147.25 | 4.38 +/- 0.11 | 0.05 +/- 0.00 |
| vllm 新底包，aipod-trans,MAX_MODEL_LEN=2048，占用约 6G 显存，Qwen3-4B-Instruct-2507-Int4-W4A16 | 0.03 +/- 0.00 | 2458.83 +/- 87.82 | 48.82 +/- 0.22 | 1328.00 +/- 143.00 | 4.39 +/- 0.05 | 0.02 +/- 0.00 |

*Values are presented as median +/- IQR (Interquartile Range). Tokenization of non-OpenAI models is approximate.*
