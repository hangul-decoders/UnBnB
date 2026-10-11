# UnBnB 참고 코드

에어비앤비체 해독기(UnBnB)의 복원 코드를 작성할 때 참고한 외부 코드 모음입니다.
원본 저장소에서 참고한 파일만 골라 **수정하지 않고** 그대로 옮겼습니다.
우리 팀이 작성한 코드와 실험 기록은 `dh_first` 브랜치에 있습니다.

## 포함한 코드

| 폴더 | 원본 저장소 | 받은 버전 (커밋) | 라이선스 | 참고한 부분 |
|---|---|---|---|---|
| `reference/Standard_Korean_GEC` | https://github.com/soyoung97/Standard_Korean_GEC | `dfe0af9` | Modified MIT (원본 README에 명시) | `gec_kr` 모델 불러오기·추론 예시(README), 학습 구조(`run.py`, `model.py`, `dataset.py`) |
| `reference/KOTOX` | https://github.com/leeyejin1231/KOTOX | `a01c917` | MIT (`LICENSE`) | 난독화 규칙 함수(`augment_funtions/`), 규칙 사전(`rules/`), 데이터 생성 흐름(`augmentation.py`) |

원본에서 옮기지 않은 것
- Standard_Korean_GEC: `KAGAS/`(교정 편집 자동 주석), `metric/`·`eval/`(GLEU·M2 평가), `get_data/`(교정 데이터 처리), `src/KoBART-gec/`(원저자가 관리 중단)
- KOTOX: `data/`, `classification/`, `finetuning/`, `evaluation/`, `assets/`, `Gwets.py`

## 참고했지만 코드는 넣지 않은 것

| 이름 | 주소 | 이유 |
|---|---|---|
| 갓종대왕 번역기 | https://github.com/thtl1999/God-sejong | LICENSE 없음 → 변형 특징만 참고 |
| KoBART | https://github.com/SKT-AI/KoBART | gec_kr의 바탕 모델 (Modified MIT) |
| gec_kr 모델 | https://huggingface.co/Soyoung97/gec_kr | 학습 출발 모델 |
| NSMC | https://github.com/e9t/nsmc | 원문 데이터 (CC0 1.0) |

## 우리 코드와의 관계 (`dh_first` 브랜치)

| 참고 코드 | 우리 코드 | 바꾼 점 |
|---|---|---|
| Standard_Korean_GEC README 추론 예시 | `part1/gec_utils.py` | 특수 토큰 중복 버그 수정, 학습 모델은 `repetition_penalty` 1.0·빔 8 |
| Standard_Korean_GEC 학습 구조 | `part3/train.py`, `part3/extend_vocab.py` | `transformers`로 새로 작성, 동적 난독화, 한글 음절 6,295자 어휘 확장 |
| KOTOX `augment_funtions/` | `part2/obfuscate.py` | 규칙 8종만 다시 구현, 버그 3개 수정(ㅋㅋ 삭제, 원문자 야민정음, 숫자 섞임), 순서 섞기 제외 |
