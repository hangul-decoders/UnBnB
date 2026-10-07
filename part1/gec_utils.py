"""gec_kr(KoBART) 불러오기와 배치 추론. 생성 설정은 Standard_Korean_GEC README 샘플 코드를 따른다."""

import torch
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast

MODEL_NAME = "Soyoung97/gec_kr"


def load(model_name=MODEL_NAME, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = PreTrainedTokenizerFast.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name).to(device).eval()
    return tok, model, device


def encode(tok, text):
    # 현재 토크나이저는 encode()가 <s>, </s>를 이미 붙인다.
    # README처럼 한 번 더 붙이면 출력이 빈 문자열이 되는 경우가 생김 (2026-10-07 확인)
    return tok.encode(text)


@torch.no_grad()
def correct(texts, tok, model, device, batch_size=64, num_beams=4, max_length=128, repetition_penalty=2.0):
    """repetition_penalty 기본값 2.0은 README 설정 (기준 모델용). 학습한 모델은 1.0으로 부른다."""
    pad = tok.pad_token_id
    outs = []
    for b in range(0, len(texts), batch_size):
        ids = [encode(tok, t)[:max_length] for t in texts[b:b + batch_size]]
        width = max(map(len, ids))
        input_ids = torch.tensor([x + [pad] * (width - len(x)) for x in ids], device=device)
        mask = (input_ids != pad).long()
        gen = model.generate(input_ids, attention_mask=mask, max_length=max_length,
                             eos_token_id=tok.eos_token_id, num_beams=num_beams,
                             early_stopping=True, repetition_penalty=repetition_penalty)
        outs += tok.batch_decode(gen, skip_special_tokens=True)
    return [o.strip() for o in outs]
