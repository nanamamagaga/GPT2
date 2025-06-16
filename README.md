Add new GPT2 distillation & LoRA modules and dataset files
- GPT2/LoRA_adapter.py
- GPT2/distillation_data_generation.py
- GPT2/data/Full_gpt4_distillation_sonnet_outputs.txt
- GPT2/data/gpt4_2line_sonnet.jsonl

실행
1. paraphrase_detection

- Colab에서 실행
링크로 들어가시면 자세히 안내되어 있습니다.<br>
https://colab.research.google.com/drive/1ZTxD1iYyzfQuDOC7B-nspRLKxk_EyWGn?usp=sharing

- 결과: train loss : 0.057 dev acc : 0.899

- 가중치
가중치 (10-1e-05-paraphrase.pt)와 생성 파일 (para-dev-output.csv,para-test-output.csv)은
용량이 커서 google drive에 올렸습니다.<br>
https://drive.google.com/drive/folders/1oTZqrA6zD7hf_rNompSz5Rwkp1AQmrOc?usp=sharing

<br>
2. sonnet_generation

- Colab에서 실행
링크로 들어가시면 자세히 안내되어 있습니다.<br>
https://colab.research.google.com/drive/1TkYc04rIGOsBEWCn3LlHrDSPuC7CmTbh?usp=sharing

- 결과: train loss : ? dev acc : ?

- 가중치
가중치 (10-1e-05-paraphrase.pt)와 생성 파일 (para-dev-output.csv,para-test-output.csv)은
용량이 커서 google drive에 올렸습니다.<br>
?
