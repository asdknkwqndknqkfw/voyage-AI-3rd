### 과제 요구 사항들을 구현하고, epoch마다의 train loss와 최종 모델의 test accuracy가 print된 notebook을 public github repository에 업로드하여 공유해주시면 됩니다. 반드시 출력 결과가 남아있어야 합니다. 
- [x] AG_News dataset 준비
	- Huggingface dataset의 `fancyzhx/ag_news`를 load
	- `collate_fn` 함수에 다음 수정사항들을 반영
    - Truncation과 관련된 부분들을 삭제
- [x] Classifier output, loss function, accuracy function 변경
	- 뉴스 기사 분류 문제는 binary classification이 아닌 일반적인 classification 문제입니다. MNIST 과제에서 했던 것 처럼 `nn.CrossEntropyLoss` 를 추가하고 `TextClassifier`의 출력 차원을 잘 조정하여 task를 풀 수 있도록 수정
	- 그리고 정확도를 재는 `accuracy` 함수도 classification에 맞춰 수정
- [x]  학습 결과 report
    - DistilBERT 실습과 같이 매 epoch 마다의 train loss를 출력하고 최종 모델의 test accuracy를 report 첨부

---
## Q1) 어떤 task를 선택하셨나요?
NER


## Q2) 모델은 어떻게 설계하셨나요? 설계한 모델의 입력과 출력 형태가 어떻게 되나요?
> 모델의 입력과 출력 형태 또는 shape을 정확하게 기술
- 모델은 어떻게 설계
  - 사전학습된 DistilBERT 모델을 encoder
  - 토큰분류기: 선형 모델 

- 설계한 모델의 입력과 출력 형태
  - 입력: [batch_size, seq_len]
  - 출력: [batch_size, 3]


## Q3) 실제로 pre-trained 모델을 fine-tuning했을 때 loss curve은 어떻게 그려지나요? 그리고 pre-train 하지 않은 Transformer를 학습했을 때와 어떤 차이가 있나요? 
- 비교 metric으로 loss curve, accuracy 비교시, epoch 증가하며 테스트 데이터에서 점차 변화를 보이지 않았습니다.

- 차이점
  - Pre-trained 모델은 이미 대규모 말뭉치에서 언어적 패턴(문법, 어휘, 의미 등)을 학습했기 때문에, fine-tuning만 해도 **높은 정확도(80~90%)**에 도달 가능

  - Pre-train 없이 학습하는 모델은 의미적 지식이 없기 때문에 MNLI에서 성능이 무작위 수준 (33%)에 가까울 수 있음

    - 큰 모델일수록 학습조차 어려움 (오버피팅, 비효율적 학습 등)


