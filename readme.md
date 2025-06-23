Developing of Llama-3-LLaMS
===
<details open>
<summary>English</summary>

# Llama-3-LLaMS
Large Language Model for SmartSocial, Inc. built with Meta Llama 3.

This project is supported and maintained by [(주)스마트소셜](http://www.smartsocial.co.kr/).

## Foundation Models Information
Meta's Llama 3 series are large language models that is freely available for use by small businesses (less than 700 million monthly active users).
### Meta-Llama-3-8B
- Base Model
- 8 billion parameters
- Text-generation tasks

### Meta-Llama-3-8B-Instruct
- Instruction Model
- 8 billion parameters
- Chat completion tasks

## Dataset
- [kowiki 데이터셋 (2024년 5월 20일자)](https://dumps.wikimedia.org/kowiki/20240520/) preprocessed by [https://github.com/lovit/kowikitext/]
    - For finetuning to Base Model

- Korean-English conversation dataset
    - Gemini-1.5-flash generates Korean-English pair conversation dataset about 4K items
    - Korean and English conversations in each pairs has same contexts
    - Data augmentation with 16K data was done as converting each conversation pairs to q_eng-a_eng, q_eng-a_kor, q_kor-a_eng, and q_kor-a_kor conversations

## Fine-tuning
This models are finetuned with [Torchtune](https://pytorch.org/torchtune/main/).
Specially, it refers the page [Meta Llama3 in torchtune](https://pytorch.org/torchtune/main/tutorials/llama3.html).

## Making Instruction Model
According to [Chat Vector](https://arxiv.org/abs/2310.04799), we prepare finetuned instruction models with finetuned base models, weight differences between original base model and instruction model.
Subtracting the weight differences to the weights of finetuned base models, we obtain models which is similar to finetuned instruction models. 
Then, we obtain finetune instruction models which is finetuned by large base data and instruction data if we finetune the similar models from high-quality instruction data.
</details>

<details>
<summary>한국어</summary>

# Llama-3-LLaMS
Meta Llama 3 기반으로 만들어진 Large Language Model for SmartSocial, Inc.

이 프로젝트는 [(주)스마트소셜](http://www.smartsocial.co.kr/)에 의해 지원·관리되고 있음.

## 기반 모델 정보
Meta's Llama 3 시리즈는 자유롭게 사용할 수 있는 공개 large language model.
### Meta-Llama-3-8B
- 기본 모델
- 약 80억개의 매개변수
- 텍스트를 입력 받아 뒤의 텍스트를 예측하여 출력하는 프로세스에 특화

### Meta-Llama-3-8B-Instruct
- Instruction 모델
- 약 80억개의 매개변수
- system, user, assistant의 역할을 부여받아, system prompt와 user input을 기반으로 적절한 assistant의 대답이 나올 수 있도록 미세조정된 모델

## Dataset
- [kowiki 데이터셋 (2024년 5월 20일자)](https://dumps.wikimedia.org/kowiki/20240520/) preprocessed by [https://github.com/lovit/kowikitext/]
    - Base 모델 학습에 이용
- 한영 대화 데이터셋
    - Gemini-1.5-flash 기반 같은 내용의 한영 대화 데이터셋 약 4,000개 생성
    - 영질문-영답, 영질문-한답, 한질문-영답, 한질문-한답 페어로 약 16,000개까지 데이터 증강

## Fine-tuning
여기 모델들은 [Torchtune](https://pytorch.org/torchtune/main/) 패키지를 활용하여 미세조정 되었다.
특히, [Meta Llama3 in torchtune](https://pytorch.org/torchtune/main/tutorials/llama3.html) 페이지를 참고하였다.

## Instruction Model 미세조정
[Chat Vector](https://arxiv.org/abs/2310.04799) 논문에 의하면 기본 모델과 instruction 모델의 각 레이어 weight차이를
미세조정된 기본 모델의 각 레이어 weight에 빼기를 하면 instruction 모델에 기본 모델을 미세조정하기 위한 데이터를 반영한 것과 유사한 결과를 도출할 수 있다. 이에 따라, 미세조정된 instruction 모델을 준비하고, 한영 대화 데이터셋을 학습하였다.

</details>

## Environments
### python
- version: 3.11.9
### Conda
#### pytorch, nvidia
- pytorch
- torchvision
- torchaudio
- pytorch-cuda==12.1
#### pip
- torchtune
