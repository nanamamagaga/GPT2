import torch

from models.gpt2 import GPT2Model

from transformers import GPT2Model as OpenAIGPT2Model
from utils import model_size_to_params

'''
내가 만든 my_gpt와 OpenAIGPT2Model을 비교하는 함수다.
마지막 layer의 hidden state들을 비교한다.
같이 거의 같으면 print("Your GPT2 implementation is correct!")
'''

def test_gpt2(model_size='gpt2'):
  # 입력 문장을 토큰 ID 형태로 표현한 sent_ids라는 2D 텐서를 생성한다. (배치 크기: 2, 길이: 8)
  sent_ids = torch.tensor([[101, 7592, 2088, 102, 0, 0, 0, 0],
                           [101, 7592, 15756, 2897, 2005, 17953, 2361, 102]])
  # attention_mask를 나타내는 텐서를 생성하여, 각 문장에서 실제 입력 부분만 마스크 1로 표시한다.
  att_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]])

  # OpenAI 모델과 자신의 모델을 모두 로드한다.
  openai_model = OpenAIGPT2Model.from_pretrained(model_size)
  my_gpt = GPT2Model.from_pretrained(model=model_size, **model_size_to_params(model_size))

  # 마지막 레이어의 hidden state는 입력된 각 토큰마다 하나씩 존재
  # 마지막 레이어의 hidden state는 최종적인 의미 표현이라고 볼 수 있어요.
  # 이 hidden states를 이용하여 최종 결과(다음 단어 예측, 문장 생성 등)를 만들어 낸다.
  my_outputs = my_gpt(sent_ids, att_mask)
  openai_outputs = openai_model(input_ids=sent_ids, attention_mask=att_mask, output_hidden_states=True).hidden_states[-1]

  att_mask = att_mask.unsqueeze(-1)
  my_outputs['last_hidden_state'] = my_outputs['last_hidden_state'] * att_mask
  openai_outputs *= att_mask

  assert torch.allclose(my_outputs['last_hidden_state'], openai_outputs, atol=1e-1, rtol=1e-2)

  print("Your GPT2 implementation is correct!")

if __name__ == '__main__':
  test_gpt2('gpt2')
