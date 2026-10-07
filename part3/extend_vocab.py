"""gec_kr 어휘에 없는 한글 음절을 추가한 토크나이저·모델을 만든다 (<unk> 대응, 2026-10-07 결정).

- 토크나이저: 문자 단위 BPE의 기본 어휘(vocab)에 빠진 음절을 직접 넣는다.
  (add_tokens로 넣으면 띄어쓰기 처리 방식이 달라지므로 BPE 어휘에 넣는 방식을 쓴다)
- 모델: 임베딩을 늘리고, 새 음절의 처음 값은 초성·중성이 같은 기존 음절들의 평균으로 둔다.
  (없으면 초성이 같은 음절 평균, 그것도 없으면 전체 음절 평균)

실행 (프로젝트 루트에서): python part3/extend_vocab.py
출력: models/gec_kr_ext/ (토크나이저 + 모델)
"""

import json
import random
import sys
from pathlib import Path

import torch
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "part2"))
from obfuscate import decompose, obfuscate  # noqa: E402

BASE = "Soyoung97/gec_kr"
OUT = ROOT / "models/gec_kr_ext"


def main():
    tok = PreTrainedTokenizerFast.from_pretrained(BASE)
    model = BartForConditionalGeneration.from_pretrained(BASE)
    unk = tok.unk_token_id

    syllables = [chr(c) for c in range(0xAC00, 0xD7A4)]
    missing = [s for s in syllables if unk in tok.encode(s, add_special_tokens=False)]
    present = {s: tok.convert_tokens_to_ids(s) for s in syllables if s not in set(missing)}
    print(f"빠진 음절 {len(missing):,} / {len(syllables):,}")

    # 1) 토크나이저: BPE 기본 어휘에 추가
    spec = json.loads(tok.backend_tokenizer.to_str())
    vocab = spec["model"]["vocab"]
    old_size = len(tok)
    next_id = max(max(vocab.values()), max(a["id"] for a in spec["added_tokens"])) + 1
    for s in missing:
        vocab[s] = next_id
        next_id += 1
    new_tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_str(json.dumps(spec)),
                                      **{k: getattr(tok, k) for k in
                                         ["bos_token", "eos_token", "unk_token", "pad_token", "mask_token"]})
    new_size = len(new_tok)
    print(f"어휘 {old_size:,} -> {new_size:,}")

    # 2) 모델 임베딩 확장 + 초기화
    model.resize_token_embeddings(new_size)
    emb = model.get_input_embeddings().weight
    by_cj, by_c = {}, {}
    for s, i in present.items():
        c, j, _ = decompose(s)
        by_cj.setdefault((c, j), []).append(i)
        by_c.setdefault(c, []).append(i)
    all_ids = list(present.values())
    with torch.no_grad():
        for s in missing:
            c, j, _ = decompose(s)
            ids = by_cj.get((c, j)) or by_c.get(c) or all_ids
            emb[new_tok.convert_tokens_to_ids(s)] = emb[ids].mean(0)
    print("출력층이 입력 임베딩과 공유됨:", model.get_output_embeddings().weight.data_ptr() == emb.data_ptr())

    # 3) 확인
    rng = random.Random(0)
    tests = ["홤장실이 넘무 작아욤", "엂다 윾머 낞 엏", "화장실이 너무 작아요. 10점!"]
    tests += [obfuscate("배우들의 명연기와 세련된 연출이 돋보이는 영화", rng)[0] for _ in range(20)]
    n_unk = sum(new_tok.encode(t, add_special_tokens=False).count(unk) for t in tests)
    roundtrip = all(new_tok.decode(new_tok.encode(t), skip_special_tokens=True) == t for t in tests)
    same_ids = all(new_tok.encode(t) == tok.encode(t) for t in ["화장실이 너무 작아요.", "연기가 좋다 10점"])
    print(f"확인: <unk> {n_unk}개, 디코딩 복원 {'OK' if roundtrip else '실패'}, "
          f"기존 글자 토큰화 동일 {'OK' if same_ids else '실패'}")
    print("예:", tests[0], "->", new_tok.convert_ids_to_tokens(new_tok.encode(tests[0])))

    OUT.mkdir(parents=True, exist_ok=True)
    new_tok.save_pretrained(OUT)
    model.save_pretrained(OUT)
    print("저장:", OUT)


if __name__ == "__main__":
    main()
