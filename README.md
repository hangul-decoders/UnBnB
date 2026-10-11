# UnBnB 참고 코드

에어비앤비체 해독기(UnBnB)를 만들기 위해 조사한 참조 코드입니다.
원본 저장소에서 참고할 파일만 골라 **수정하지 않고** 그대로 옮겼습니다.

- 아이디어: 숙소 후기를 번역기가 못 읽도록 일부러 비틀어 쓴 한국어("홤장실이 넘무 작아욤")를 표준 한국어("화장실이 너무 작아요")로 복원한다.
- 방법: 한국어 문법 교정 모델(KoBART 기반 `gec_kr`)에 (난독문, 원문) 쌍을 추가 학습시킨다. 난독문은 난독화 규칙으로 직접 만든다.

## 1. 참조할 코드 조사

| 폴더 | 원본 저장소 | 받은 버전 (커밋) | 라이선스 | 무엇을 참고하나 |
|---|---|---|---|---|
| `reference/Standard_Korean_GEC` | https://github.com/soyoung97/Standard_Korean_GEC (ACL 2023) | `dfe0af9` | Modified MIT (원본 README에 명시) | KoBART로 "틀린 문장 → 고친 문장"을 학습하는 코드, 공개 모델 `gec_kr` 사용법 |
| `reference/KOTOX` | https://github.com/leeyejin1231/KOTOX (arXiv 2025) | `a01c917` | MIT (`LICENSE`) | 한국어 난독화 규칙 함수 (받침 추가, 연음, 된소리, 야민정음 등) |

원본에서 옮기지 않은 것
- Standard_Korean_GEC: `KAGAS/`(교정 편집 자동 주석), `metric/`·`eval/`(GLEU·M2 평가), `get_data/`(교정 데이터 처리), `src/KoBART-gec/`(원저자가 관리 중단)
- KOTOX: `data/`, `classification/`, `finetuning/`, `evaluation/`, `assets/`, `Gwets.py`

참고만 하고 코드는 넣지 않은 것

| 이름 | 주소 | 이유 |
|---|---|---|
| 갓종대왕 번역기 | https://github.com/thtl1999/God-sejong | 에어비앤비체 생성기. LICENSE 없음 → 변형 특징만 참고 |
| KoBART | https://github.com/SKT-AI/KoBART | gec_kr의 바탕 모델 |
| gec_kr 모델 | https://huggingface.co/Soyoung97/gec_kr | 출발 모델 |

## 2. 참조 코드 분석

### Standard_Korean_GEC (학습 코드)

| 파일 | 하는 일 |
|---|---|
| `dataset.py` `KoBARTGecDataset` | 한 줄에 `틀린 문장\t고친 문장`인 파일을 읽는다. `__getitem__`에서 입력(`input_ids`), 디코더 입력(`decoder_input_ids` = 시작 토큰 + 정답 앞부분), 정답(`labels`, 빈 칸은 -100으로 손실 계산에서 제외)을 만든다 |
| `model.py` `KoBARTConditionalGeneration` | `BartForConditionalGeneration`을 감싼 PyTorch Lightning 모듈. `forward`가 입력·디코더 입력·정답을 넣어 손실을 받고, `training_step`이 그 손실로 학습한다. `generate`는 빔 4로 문장을 만든다 |
| `run.py` | 인자(학습률 3e-5, 에폭 등)를 받아 학습·평가를 실행 |
| `README.md` | 공개 모델 `gec_kr` 추론 예시: 토크나이저로 문장을 숫자로 바꾸고 → `generate`(빔 4, `repetition_penalty` 2.0) → 다시 글자로 |

→ 데이터 파일만 `난독문\t원문`으로 바꾸면 같은 구조로 학습할 수 있다.

### KOTOX (난독화 규칙)

| 파일 | 주요 함수 | 에어비앤비체 특징 |
|---|---|---|
| `augment_funtions/phonetic_addition.py` | `phonological_addition_adaptive_final_consonant`, `..._final_consonant` | 받침 추가·중복 (너무 → 넘무) |
| | `phonological_addition_initial_consonant` | 초성 추가 (실이 → 실리) |
| | `phonological_addition_semivowel` | 반모음 첨가 (너무 → 녀무) |
| `augment_funtions/processing.py` | `continue_sound` | 연음 (좋아요 → 조아요) |
| | `first_power_replace` | 된소리·거센소리 (기대 → 끼대) |
| | `vowel_replace`, `last_replace` | 모음 대치(ㅐ↔ㅔ), 받침 대치 |
| `augment_funtions/rule.py` | `yamin_swap` | 야민정음 (대박 → 머박) |
| | `change_array` | 순서 섞기 |
| `augmentation.py` | — | 위 함수를 묶어 데이터를 만드는 실행 파일 (유해 표현 데이터용) |

함수마다 어절(띄어쓰기 단위) 하나를 받아 변형된 어절을 돌려준다. 한 문장에 규칙 2~4개를 적용한다.

## 3. 사용할 NLP 모델의 원리: BART (KoBART)

- **구조**: 인코더-디코더 트랜스포머. 인코더(6층)가 입력 문장 전체를 양방향으로 읽고, 디코더(6층)가 앞에 쓴 토큰과 인코더 표현(크로스 어텐션)을 보며 다음 토큰을 하나씩 만든다.
- **사전학습**: 문장 일부를 가리거나 망가뜨린 뒤 원문으로 복원하도록 학습한다(노이즈 복원). "망가진 문장 → 원문"이라는 점이 에어비앤비체 복원과 같다.
- **미세조정**: `dataset.py`처럼 디코더에 정답 앞부분을 넣고(teacher forcing) 다음 토큰의 교차 엔트로피 손실을 줄인다 (`model.py`의 `outs.loss`).
- **생성**: 빔 서치로 후보 문장 여러 개를 이어 쓰며 확률이 가장 높은 문장을 고른다 (`generate`, `num_beams`).
