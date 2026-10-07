from __future__ import annotations
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from common.data import repo_path
from common.metrics import masked_mean


def sha(path):
    return hashlib.sha256(repo_path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def append(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')


def records(path):
    path = Path(path)
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []


def policy_scores(model, generated, entropy=False):
    seq = generated['sequences']
    attn = generated['attention_mask']
    positions = (attn.cumsum(-1) - 1).clamp_min(0)
    outputs = model(input_ids=seq, attention_mask=attn,
                    position_ids=positions, use_cache=False)
    width = generated['prompt_width']
    logits = outputs.logits[:, width-1:-1, :]
    ids = generated['response_ids']
    chunks, entropies = [], []
    for start in range(0, ids.shape[1], 32):
        scores = logits[:, start:start+32].float()
        labels = ids[:, start:start+32]
        logp = -F.cross_entropy(scores.reshape(-1, scores.shape[-1]),
                                labels.reshape(-1), reduction='none')
        chunks.append(logp.reshape(labels.shape))
        if entropy:
            with torch.no_grad():
                lp = F.log_softmax(scores, -1)
                entropies.append(-(lp.exp() * lp).sum(-1))
    token_logp = torch.cat(chunks, -1)
    token_entropy = torch.cat(entropies, -1) if entropy else None
    return token_logp, token_entropy


def critic_scores(model, generated):
    # Access the underlying sequence-classification model while retaining its
    # injected LoRA layers and PEFT modules-to-save value head.
    base = model.get_base_model() if hasattr(model, 'get_base_model') else model
    backbone = getattr(base, base.base_model_prefix)
    attn = generated['attention_mask']
    hidden = backbone(input_ids=generated['sequences'], attention_mask=attn,
                      position_ids=(attn.cumsum(-1)-1).clamp_min(0),
                      use_cache=False, return_dict=True).last_hidden_state
    head = base.score if hasattr(base, 'score') else base.classifier
    width = generated['prompt_width']
    # V(s_t) is evaluated BEFORE response token a_t, matching its log-probability.
    return head(hidden[:, width-1:-1, :]).squeeze(-1).float()


def normal_tensors(generated):
    # generate() returns inference tensors; clone before autograd uses indices.
    return {k: v.clone() if torch.is_tensor(v) else v
            for k, v in generated.items()}


def trainable_state(model):
    return {n: p.detach().cpu().clone() for n,p in model.named_parameters()
            if p.requires_grad}


def restore_trainable(model, state):
    with torch.no_grad():
        parameters = dict(model.named_parameters())
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name].device))


def cuda_required():
    if not torch.cuda.is_available():
        raise RuntimeError('This command requires a GPU runtime.')
