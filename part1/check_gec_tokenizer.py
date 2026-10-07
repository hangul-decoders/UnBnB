"""Part 1 ③ gec_kr에 난독문 넣어 보기, ④ KoBART 토크나이저 <unk> 확인.

실행 (프로젝트 루트에서): python part1/check_gec_tokenizer.py
출력: part1/gec_tokenizer_결과.md
"""

import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "part1"))
sys.path.insert(0, str(ROOT / "part2"))
from gec_utils import load, correct  # noqa: E402
from obfuscate import obfuscate  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

# 계획서 예시 + 직접 만든 숙소 후기 예시
HANDMADE = [
    ("홤장실이 넘무 작아욤", "화장실이 너무 작아요"),
    ("연끼가 넘무 조아욤", "연기가 너무 좋아요"),
    ("내용이 넘무 지루햇어욤", "내용이 너무 지루했어요"),
    ("숙쏘까 넘무 더러웟써요", "숙소가 너무 더러웠어요"),
    ("침때에서 냄쌔갉 낫써욤", "침대에서 냄새가 났어요"),
    ("호스트갉 친쩔하지 아낫써요", "호스트가 친절하지 않았어요"),
    ("뜨거운 뭉리 앙 나와욤", "뜨거운 물이 안 나와요"),
    ("위취는 조은데 방음이 하낭또 앙대요", "위치는 좋은데 방음이 하나도 안 돼요"),
]


def load_nsmc(path, n, seed):
    lines = path.read_text(encoding="utf-8").splitlines()[1:]
    sents = [ln.split("\t")[1].strip() for ln in lines if ln.count("\t") == 2]
    sents = [re.sub(r"\s+", " ", s) for s in sents]
    sents = [s for s in sents if len(re.findall(r"[가-힣]", s)) >= 5 and len(s) <= 60]
    sents = list(dict.fromkeys(sents))
    return random.Random(seed).sample(sents, n)


def unk_stats(tok, texts):
    unk = tok.unk_token_id
    n_tok = n_unk = n_sent = 0
    chars = Counter()
    for t in texts:
        enc = tok(t, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        n_tok += len(ids)
        hit = [o for i, o in zip(ids, offs) if i == unk]
        n_unk += len(hit)
        n_sent += bool(hit)
        for s, e in hit:
            chars[t[s:e]] += 1
    return n_tok, n_unk, n_sent, chars


def main():
    out = ["# Part 1 ③④ 결과: gec_kr 난독문 입력, KoBART 토크나이저 확인", ""]

    # ④ 토크나이저
    rng = random.Random(7)
    origs = load_nsmc(ROOT / "data/raw/ratings_train.txt", 2000, seed=7)
    obfs = [obfuscate(s, rng)[0] for s in origs]
    out += ["## ④ 토크나이저 `<unk>` 확인", "",
            "NSMC 2,000문장(원문)과 같은 문장에 난독화 규칙을 적용한 2,000문장(난독문)을 토큰화", "",
            "| 토크나이저 | 대상 | 토큰 수 | `<unk>` 수 | `<unk>` 비율 | `<unk>` 포함 문장 |",
            "|---|---|---|---|---|---|"]
    unk_chars = {}
    for name in ["Soyoung97/gec_kr", "gogamza/kobart-base-v2"]:
        tok = PreTrainedTokenizerFast.from_pretrained(name)
        for label, texts in [("원문", origs), ("난독문", obfs), ("직접 만든 예시", [h[0] for h in HANDMADE])]:
            n_tok, n_unk, n_sent, chars = unk_stats(tok, texts)
            out.append(f"| {name} | {label} | {n_tok:,} | {n_unk:,} | {n_unk / n_tok:.2%} | "
                       f"{n_sent:,}/{len(texts):,} ({n_sent / len(texts):.1%}) |")
            unk_chars[(name, label)] = chars
    out.append("")
    for (name, label), chars in unk_chars.items():
        if chars and label != "직접 만든 예시":
            top = ", ".join(f"`{c}` {n}" for c, n in chars.most_common(30))
            out += [f"- {name} / {label} `<unk>` 글자 상위: {top}"]

    tok, model, device = load()
    ex = HANDMADE[0][0]
    pieces = tok.convert_ids_to_tokens(tok.encode(ex))
    out += ["", f"토큰화 예시 (gec_kr): `{ex}` → `{' | '.join(pieces)}`", ""]

    # ③ gec_kr 실행
    preds = correct([h[0] for h in HANDMADE], tok, model, device)
    out += ["## ③ 기존 교정 모델(gec_kr)에 난독문 입력", "", f"장치: {device}", "",
            "### 직접 만든 예시", "", "| 입력(난독문) | gec_kr 출력 | 정답 | 일치 |", "|---|---|---|---|"]
    for (obf, gold), p in zip(HANDMADE, preds):
        out.append(f"| {obf} | {p} | {gold} | {'O' if p == gold else 'X'} |")

    sample_o, sample_x = origs[:30], obfs[:30]
    preds_x = correct(sample_x, tok, model, device)
    em = sum(p == g for p, g in zip(preds_x, sample_o))
    out += ["", f"### 규칙으로 만든 난독문 30개 (원문과 완전 일치 {em}/30)", "",
            "| 입력(난독문) | gec_kr 출력 | 원문 |", "|---|---|---|"]
    esc = lambda t: t.replace("|", "\\|")
    for x, p, g in zip(sample_x, preds_x, sample_o):
        out.append(f"| {esc(x)} | {esc(p)} | {esc(g)} |")

    (ROOT / "part1/gec_tokenizer_결과.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
