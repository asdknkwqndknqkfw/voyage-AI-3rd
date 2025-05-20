1. [loss](https://wandb.ai/wltkqdl-voice-song/hanghae99-week8?nw=nwuserwltkqdlvoice&panelDisplayName=train%2Floss&panelSectionName=train)

2. [학습 속도]
  - r8
    - 6m 40s
  - r128
    - 6m 51s
  - r256
    - 7m 5s
  - rank↑ ~ 학습 속도↑

3. 메모리 점유율
  - r8
    - Max Memory=4.9GB
  - r128
    - Max Memory=5.1GB
  - r256
    - Max Memory=5.3GB
  - rank↑ ~ 메모리 점유율↑

4. LoRA의 장단점 분석
장
  - rank↓ ~ 메모리 효율성↑
  - rank↓ 성능을 증가시킬 수 있다.
단
  - rank 너무 낮으면, 성능↓
  - rank 너무 높으면, 효율성↓(학습 속도↓, 메모리 점유율↑, 오히려 성능↓ 가능성 o)