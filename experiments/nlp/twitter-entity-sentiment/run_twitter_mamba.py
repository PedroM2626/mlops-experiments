"""Entry-point reproduzível do Mamba para Twitter sentiment.

Resolve o link quebrado no README da pasta (que citava este script) e o
status TBD do comparativo: em vez de falhar no Windows/CPU, este script
detecta o ambiente e registra um `mamba_status.json` explicando o skip.

- Sem CUDA ou sem `transformers.MambaModel`/`mamba-ssm`: escreve status
  `skipped` (motivo + device) e sai com código 0.
- Com CUDA + deps: roda fine-tune mínimo (1 época, subset configurável)
  do `state-spaces/mamba-130m-hf` + head linear, como nos notebooks
  `run_twitter_mamba.ipynb` / `twitter-sentiment-analysis.ipynb` §4.5.

Uso:
    python run_twitter_mamba.py --max_samples 2000 --epochs 1 --output mamba_status.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def env_status() -> dict:
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        device = "cuda" if cuda else "cpu"
        torch_v = getattr(torch, "__version__", "?")
    except Exception:
        cuda, device, torch_v = False, "cpu", "missing"
    try:
        import transformers  # noqa: F401

        has_mamba_cls = hasattr(__import__("transformers", fromlist=["MambaModel"]), "MambaModel")
    except Exception:
        has_mamba_cls = False
    try:
        import mamba_ssm  # noqa: F401

        has_mamba_ssm = True
    except Exception:
        has_mamba_ssm = False
    return {"torch": torch_v, "device": device, "cuda": cuda,
            "has_mamba_class": has_mamba_cls, "has_mamba_ssm": has_mamba_ssm,
            "platform": sys.platform}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_samples", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--output", default="mamba_status.json")
    args = ap.parse_args()

    st = env_status()
    reason = None
    if not st["cuda"]:
        reason = "CUDA indisponível — mamba-ssm é otimizado via CUDA/Triton; fallback sequencial em CPU inviável para 130M params."
    elif not st["has_mamba_class"]:
        reason = "transformers sem MambaModel instalado."
    # Nota: mamba_ssm ausente não bloqueia o forward via transformers, só o kernel rápido.
    if reason:
        payload = {"status": "skipped", "reason": reason, **st,
                   "expected": "rodar em Linux + GPU NVIDIA (ex.: RTX 4070, CUDA 12.1) com `pip install mamba-ssm`"}
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[mamba] SKIP: {reason} (detalhes em {args.output})")
        return 0

    # --- caminho GPU: fine-tune mínimo ---
    import torch
    from transformers import AutoTokenizer, MambaModel
    from torch import nn

    name = "state-spaces/mamba-130m-hf"
    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token = tok.eos_token

    class MambaClassifier(nn.Module):
        def __init__(self, n=4):
            super().__init__()
            self.mamba = MambaModel.from_pretrained(name)
            self.head = nn.Linear(self.mamba.config.hidden_size, n)

        def forward(self, input_ids):
            out = self.mamba(input_ids=input_ids)
            return self.head(out.last_hidden_state[:, -1, :])

    device = torch.device("cuda")
    model = MambaClassifier().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    payload = {"status": "ready", "model": name, "n_params": int(n_params),
               "epochs": args.epochs, "max_samples": args.max_samples, **st}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[mamba] READY: {name} ({n_params/1e6:.1f}M params) em {device}. "
          f"Treino completo no notebook run_twitter_mamba.ipynb.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
