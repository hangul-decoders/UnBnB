"""에어비앤비체 난독화 규칙.

KOTOX(https://github.com/leeyejin1231/KOTOX, MIT License, Copyright (c) 2025 Yejin Lee)의
규칙과 사전(rules/replace.json, rules/iconic_dictionary.json)을 참고해 다시 구현했다.
KOTOX 원본과 다른 점:
  - 자모만 있는 글자(ㅋㅋ, ㅠㅠ)를 지우지 않는다 (KOTOX 1-1 버그)
  - 야민정음은 한글 -> 한글 매핑만 쓴다 (원문자 ㉵ 등 제외)
  - 순서 섞기는 한글로만 된 어절에만 적용한다 (숫자·기호 보호)
  - 반모음·받침 추가 등은 어절 전체가 아니라 글자별 확률로 적용한다
  - 연음에서 ㅎ 받침은 탈락시킨다 (좋아요 -> 조아요)
  - 받침 추가에서 뒷글자 초성을 그대로 받침으로 쓰는 경우도 허용한다 (화장실 -> 홪장실)
  - 모든 무작위 선택은 인자로 받은 random.Random으로 한다 (재현 가능)
"""

import random

CHO = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
JUNG = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
JONG = [""] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")


def is_syllable(ch):
    return "가" <= ch <= "힣"


def decompose(ch):
    code = ord(ch) - 0xAC00
    return CHO[code // 588], JUNG[(code % 588) // 28], JONG[code % 28]


def compose(cho, jung, jong=""):
    return chr(0xAC00 + (CHO.index(cho) * 21 + JUNG.index(jung)) * 28 + JONG.index(jong))


# ---------- 사전 (KOTOX replace.json / iconic_dictionary.json 기반) ----------

POWER = {"ㄱ": ["ㄲ", "ㄲ", "ㅋ"], "ㄷ": ["ㄸ", "ㄸ", "ㅌ"], "ㅂ": ["ㅃ", "ㅃ", "ㅍ"],
         "ㅅ": ["ㅆ"], "ㅈ": ["ㅉ", "ㅉ", "ㅊ"]}  # 된소리 쪽에 가중치

VOWEL = {"ㅐ": ["ㅔ"], "ㅔ": ["ㅐ"], "ㅚ": ["ㅙ", "ㅞ"], "ㅙ": ["ㅚ", "ㅞ"], "ㅞ": ["ㅚ", "ㅙ"]}

SEMIVOWEL = {"ㅏ": ["ㅑ", "ㅘ"], "ㅓ": ["ㅕ", "ㅝ"], "ㅗ": ["ㅛ"], "ㅜ": ["ㅠ"], "ㅡ": ["ㅢ"], "ㅣ": ["ㅟ"]}

# 받침 대표음이 같은 그룹 (받침 대치용)
SOUND_GROUPS = [
    ["ㄱ", "ㅋ", "ㄲ", "ㄳ", "ㄺ"],
    ["ㄴ", "ㄵ", "ㄶ"],
    ["ㄷ", "ㅅ", "ㅈ", "ㅊ", "ㅌ", "ㅎ", "ㅆ"],
    ["ㄹ", "ㄼ", "ㄽ", "ㄾ", "ㅀ"],
    ["ㅁ", "ㄻ"],
    ["ㅂ", "ㅍ", "ㄿ", "ㅄ"],
]
SAME_SOUND = {j: [x for x in g if x != j] for g in SOUND_GROUPS for j in g}

# 뒷글자 초성 -> 앞글자에 붙일 받침 (발음 기준)
INIT_TO_FINAL = {"ㄱ": "ㄱ", "ㄴ": "ㄴ", "ㄷ": "ㄷ", "ㄹ": "ㄹ", "ㅁ": "ㅁ", "ㅂ": "ㅂ", "ㅅ": "ㅅ", "ㅇ": "ㅇ",
                 "ㅈ": "ㄷ", "ㅊ": "ㄷ", "ㅌ": "ㄷ", "ㅍ": "ㅂ", "ㅋ": "ㄱ", "ㅎ": "ㅇ",
                 "ㄲ": "ㄱ", "ㄸ": "ㄷ", "ㅃ": "ㅂ", "ㅆ": "ㅅ", "ㅉ": "ㄷ"}
# 단어 끝 글자에 붙이는 받침 (조아욤, 감사함당 같은 형태)
END_FINALS = ["ㅁ", "ㅁ", "ㅇ", "ㅇ", "ㄴ", "ㄱ", "ㅂ"]

# 앞글자 받침 -> 뒷글자 ㅇ 자리에 넣을 초성 (초성 추가용)
FINAL_TO_INIT = {"ㄱ": "ㄱ", "ㄴ": "ㄴ", "ㄷ": "ㄷ", "ㄹ": "ㄹ", "ㅁ": "ㅁ", "ㅂ": "ㅂ", "ㅅ": "ㅅ",
                 "ㄶ": "ㅎ", "ㅀ": "ㅎ", "ㄳ": "ㅅ", "ㄵ": "ㅈ", "ㄺ": "ㄱ", "ㄻ": "ㅁ", "ㄼ": "ㅂ",
                 "ㄽ": "ㅅ", "ㄾ": "ㄷ", "ㄿ": "ㅂ", "ㅄ": "ㅅ"}

# 연음: 앞글자 받침 -> (남는 받침, 뒷글자 초성). 뒷글자 초성이 ㅇ일 때만
LIAISON = {"ㄱ": ("", "ㄱ"), "ㄴ": ("", "ㄴ"), "ㄷ": ("", "ㄷ"), "ㄹ": ("", "ㄹ"), "ㅁ": ("", "ㅁ"),
           "ㅂ": ("", "ㅂ"), "ㅅ": ("", "ㅅ"), "ㅈ": ("", "ㅈ"), "ㅊ": ("", "ㅊ"), "ㅋ": ("", "ㅋ"),
           "ㅌ": ("", "ㅌ"), "ㅍ": ("", "ㅍ"), "ㅎ": ("", "ㅇ"), "ㄲ": ("", "ㄲ"), "ㅆ": ("", "ㅆ"),
           "ㄳ": ("ㄱ", "ㅆ"), "ㄵ": ("ㄴ", "ㅈ"), "ㄶ": ("", "ㄴ"), "ㄺ": ("ㄹ", "ㄱ"), "ㄻ": ("ㄹ", "ㅁ"),
           "ㄼ": ("ㄹ", "ㅂ"), "ㄽ": ("ㄹ", "ㅆ"), "ㄾ": ("ㄹ", "ㅌ"), "ㄿ": ("ㄹ", "ㅍ"), "ㅀ": ("", "ㄹ"),
           "ㅄ": ("ㅂ", "ㅆ")}

# 야민정음 (KOTOX yamin_dict 중 한글 -> 한글 항목)
YAMIN = {"파": "과", "과": "파", "머": "대", "대": "머", "귀": "커", "커": "귀", "팡": "광", "광": "팡",
         "관": "판", "판": "관", "팔": "괄", "괄": "팔", "비": "네", "네": "비", "멍": "댕", "댕": "멍",
         "면": "띤", "명": "띵", "유": "윾", "우": "윽", "점": "겸", "피": "괴", "괴": "피", "퍼": "떠",
         "떠": "퍼", "공": "끙", "의": "익"}


# ---------- 글자 단위 규칙: (word, rng, p) -> word ----------

def _per_char(word, rng, p, fn):
    """fn(chars, i) 가 바꿀 글자(들)를 반환하면 확률 p로 적용. 최소 1곳은 바꾼다."""
    chars = list(word)
    cands = [i for i in range(len(chars)) if is_syllable(chars[i]) and fn(chars, i) is not None]
    if not cands:
        return word
    picked = [i for i in cands if rng.random() < p] or [rng.choice(cands)]
    for i in sorted(picked, reverse=True):
        res = fn(chars, i)
        if res is not None:
            for k, v in res.items():
                chars[k] = v
    return "".join(chars)


def _next_syl(chars, i):
    return decompose(chars[i + 1]) if i + 1 < len(chars) and is_syllable(chars[i + 1]) else None


def r_power(word, rng, p=0.5):
    """된소리·거센소리: 번역 -> 뻔역"""
    def fn(chars, i):
        c, v, j = decompose(chars[i])
        return {i: compose(rng.choice(POWER[c]), v, j)} if c in POWER else None
    return _per_char(word, rng, p, fn)


def r_vowel(word, rng, p=0.6):
    """모음 대치: 대박 -> 데박"""
    def fn(chars, i):
        c, v, j = decompose(chars[i])
        return {i: compose(c, rng.choice(VOWEL[v]), j)} if v in VOWEL else None
    return _per_char(word, rng, p, fn)


def r_final_swap(word, rng, p=0.5):
    """받침 대치 (소리가 같은 받침으로): 있어요 -> 잇어요"""
    def fn(chars, i):
        c, v, j = decompose(chars[i])
        return {i: compose(c, v, rng.choice(SAME_SOUND[j]))} if j in SAME_SOUND else None
    return _per_char(word, rng, p, fn)


def r_semivowel(word, rng, p=0.3):
    """반모음 첨가: 너무 -> 녀무"""
    def fn(chars, i):
        c, v, j = decompose(chars[i])
        return {i: compose(c, rng.choice(SEMIVOWEL[v]), j)} if v in SEMIVOWEL else None
    return _per_char(word, rng, p, fn)


def r_add_final(word, rng, p=0.4):
    """받침 추가·중복: 너무 -> 넘무, 화장실 -> 홪장실/홛장실, 좋아요 -> 좋아욤"""
    def fn(chars, i):
        c, v, j = decompose(chars[i])
        if j:
            return None
        nxt = _next_syl(chars, i)
        if nxt is None:
            return {i: compose(c, v, rng.choice(END_FINALS))}
        n_cho = nxt[0]
        options = [INIT_TO_FINAL[n_cho]]
        if n_cho in JONG and n_cho != INIT_TO_FINAL[n_cho]:
            options.append(n_cho)  # 뒷글자 초성을 그대로 받침으로 (받침 중복)
        return {i: compose(c, v, rng.choice(options))}
    return _per_char(word, rng, p, fn)


def r_add_initial(word, rng, p=0.6):
    """초성 추가: 실이 -> 실리, 작아 -> 작가"""
    def fn(chars, i):
        c, v, j = decompose(chars[i])
        if c != "ㅇ" or i == 0 or not is_syllable(chars[i - 1]):
            return None
        pj = decompose(chars[i - 1])[2]
        return {i: compose(FINAL_TO_INIT[pj], v, j)} if pj in FINAL_TO_INIT else None
    return _per_char(word, rng, p, fn)


def r_liaison(word, rng, p=0.7):
    """연음: 작아요 -> 자가요, 좋아요 -> 조아요"""
    def fn(chars, i):
        c, v, j = decompose(chars[i])
        nxt = _next_syl(chars, i)
        if not j or nxt is None or nxt[0] != "ㅇ" or j not in LIAISON:
            return None
        keep, new_cho = LIAISON[j]
        return {i: compose(c, v, keep), i + 1: compose(new_cho, nxt[1], nxt[2])}
    return _per_char(word, rng, p, fn)


def r_yamin(word, rng, p=0.8):
    """야민정음: 명작 -> 띵작, 대박 -> 머박"""
    def fn(chars, i):
        return {i: YAMIN[chars[i]]} if chars[i] in YAMIN else None
    return _per_char(word, rng, p, fn)


def r_shuffle(word, rng, p=None):
    """순서 섞기 (한글로만 된 3글자 이상 어절): 캠브릿지 -> 캠릿브지"""
    if len(word) < 3 or not all(is_syllable(ch) for ch in word):
        return word
    chars = list(word)
    if len(chars) == 3:
        chars[1], chars[2] = chars[2], chars[1]
    else:
        mid = chars[1:-1]
        for _ in range(5):
            rng.shuffle(mid)
            if mid != chars[1:-1]:
                break
        chars = [chars[0]] + mid + [chars[-1]]
    return "".join(chars)


# 적용 순서대로 나열 (연음·초성 추가가 받침 추가보다 먼저, 순서 섞기는 마지막)
# (이름, 함수, 뽑힐 가중치, 문장 내 적용 어절 비율)
RULES = [
    ("연음", r_liaison, 3, 0.4),
    ("초성추가", r_add_initial, 1, 0.25),
    ("받침추가", r_add_final, 4, 0.4),
    ("된소리", r_power, 3, 0.4),
    ("모음대치", r_vowel, 2, 0.4),
    ("받침대치", r_final_swap, 2, 0.4),
    ("반모음", r_semivowel, 1, 0.25),
    ("야민정음", r_yamin, 1, 0.5),
    ("순서섞기", r_shuffle, 1, 0.2),
]


def obfuscate(sentence, rng, n_rules=None, max_tries=10):
    """문장에 규칙 2~4개를 골라 적용. (난독문, 적용 규칙 이름 리스트) 반환.
    고른 규칙이 하나도 안 걸려 원문 그대로면 다시 고른다."""
    for _ in range(max_tries):
        obf, applied = _obfuscate_once(sentence, rng, n_rules)
        if obf != sentence:
            break
    return obf, applied


def _obfuscate_once(sentence, rng, n_rules):
    words = sentence.split(" ")
    if n_rules is None:
        n_rules = rng.choices([2, 3, 4], weights=[4, 4, 2])[0]
    names = [r[0] for r in RULES]
    weights = [r[2] for r in RULES]
    chosen = set()
    while len(chosen) < min(n_rules, len(RULES)):
        chosen.add(rng.choices(names, weights=weights)[0])

    applied = []
    for name, fn, _, ratio in RULES:
        if name not in chosen:
            continue
        cands = [k for k, w in enumerate(words) if fn(w, random.Random(0)) != w]
        if not cands:
            continue
        n = max(1, round(len(words) * ratio))
        for k in rng.sample(cands, min(n, len(cands))):
            words[k] = fn(words[k], rng)
        applied.append(name)
    return " ".join(words), applied


def check_pair(obf, orig):
    """자동 검사. 문제 목록을 반환 (비어 있으면 통과)."""
    problems = []
    if obf == orig:
        problems.append("변화 없음")
    if len(obf.split(" ")) != len(orig.split(" ")):
        problems.append("어절 수 다름")
    keep = lambda s: [ch for ch in s if not is_syllable(ch)]  # 숫자·기호·자모·영문·공백
    if keep(obf) != keep(orig):
        problems.append("한글 음절 외 문자 변경")
    if len(obf) != len(orig):
        problems.append("글자 수 다름")
    return problems
