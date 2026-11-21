"""The ONLY script that runs outside the sandbox (for the Apple GPU, MPS). Keep it small and auditable.

Run exactly as (from the project folder, no `cd` in the same command):
    .venv/bin/python src/gpu_run.py              -> prints whether MPS is available
    .venv/bin/python src/gpu_run.py embed        -> sentence embeddings of description_day0 (dev only)
    .venv/bin/python src/gpu_run.py finetune grid               -> 4 fixed settings x 3 periods, seed 0
    .venv/bin/python src/gpu_run.py finetune one <setting> [...] -> given setting(s) x 3 periods, seed 0
    .venv/bin/python src/gpu_run.py finetune seeds <setting>    -> chosen setting, seed 1 (2 seeds in total)
    .venv/bin/python src/gpu_run.py embed-variant <name>        -> MiniLM embeddings of edited DEV texts
    .venv/bin/python src/gpu_run.py embed-test                  -> MiniLM embeddings of TEST texts (Phase 6, step D)

What `embed-test` does, and nothing else (Phase 6 step D; run only with the user's explicit OK):
  READS  data/test_locked.parquet, columns cve_id + description_day0 ONLY (no labels, no other columns), and the
         MiniLM revision recorded in data/embeddings/minilm_meta.json; model files from .cache/hf, OFFLINE
         (HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1). Refuses to run without MPS.
  WRITES data/embeddings/test/minilm.npy (float16, L2-normalised, test order) + minilm_ids.parquet + minilm_meta.json
         data/embeddings/test/phrase_removed_input.parquet (cve_id, text) + phrase_removed_minilm.npy: for EVERY
         test row whose description_day0 contains an exploitation phrase, the text with the phrase clause removed
         (description_leak.drop_sentences_with) - used by the uncertain-phrase sensitivity (picked in phase6_test.py,
         so this script never needs the desc_day0_uncertain column).
  Same encoder settings as `embed` (max_seq_length 256, batch 128, normalize_embeddings=True).

What `embed-variant <name>` does, and nothing else (Step 7 sensitivity checks):
  READS  data/embeddings/variants/<name>_input.parquet (cve_id, text; written by a sandboxed script) and the
         cve_id list of data/dev.parquet (guard: refuses any cve_id that is not a dev CVE)
  LOADS  MiniLM at the revision recorded in data/embeddings/minilm_meta.json, OFFLINE (HF_HUB_OFFLINE=1)
  WRITES data/embeddings/variants/<name>_minilm.npy (float16, L2-normalised, input order) + _meta.json

What `embed` does, and nothing else:
  READS  data/dev.parquet, columns cve_id + description_day0 only (train+val; never test_locked.parquet)
  DOWNLOADS the two models below from Hugging Face into .cache/hf (inside the project), pinned to the
         commit hash resolved at run time (recorded in _meta.json as "revision")
  WRITES data/embeddings/<model>.npy (float16, L2-normalised, one row per CVE in dev order),
         data/embeddings/<model>_ids.parquet (cve_id order) and data/embeddings/<model>_meta.json

What `finetune` does, and nothing else (Step 6c):
  READS  data/dev.parquet (cve_id, published, split, label, description_day0) and the MiniLM revision
         recorded in data/embeddings/minilm_meta.json; the model files come from .cache/hf
  TRAINS MiniLM + a linear head with class-weighted BCE, per period: epoch search (max 3) on the training
         period minus its last 6 months, stopping set = those 6 months; then refit on the full period for the
         best epoch count (same seed); scores the period's evaluation rows. Choosing a setting is done later,
         in the sandboxed notebook - this script only trains and scores.
  WRITES data/finetune/<setting>_s<seed>_<period>.parquet (cve_id, score) and .json (loss history, timing)
  Speed: negatives in the training and stopping sets are subsampled to NEG_FRAC (all positives kept; the
  class weight is computed on the subsample) with ONE fixed subsample seed, so every setting and every training
  seed sees exactly the same CVEs. Evaluation rows are always scored in full.
  Offline: HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 are set, so `finetune` makes no network calls (the pinned
  model is already in .cache/hf). The .json per run holds per-epoch training and stopping-set loss.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# keep every cache inside the project (CLAUDE.md), overriding any machine-wide setting; before importing torch
os.environ["HF_HOME"] = str(ROOT / ".cache" / "hf")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(ROOT / ".cache" / "hf")
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache" / "mpl")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch  # noqa: E402

MODELS = {   # short name -> Hugging Face id
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",   # 384-d, fast baseline encoder
    "bge_small": "BAAI/bge-small-en-v1.5",                 # 384-d, stronger on retrieval/classification benchmarks
}
INPUT = ROOT / "data" / "dev.parquet"
OUT_DIR = ROOT / "data" / "embeddings"
MAX_SEQ_LEN = 256          # CVE descriptions: median ~260 characters (~60 tokens); 256 tokens covers almost all
BATCH = 128


def embed() -> None:
    import json
    import time

    import numpy as np
    import pandas as pd
    import sentence_transformers
    from huggingface_hub import HfApi
    from sentence_transformers import SentenceTransformer

    assert INPUT.name == "dev.parquet"                     # guard: this script never reads the test set
    df = pd.read_parquet(INPUT, columns=["cve_id", "description_day0"])
    texts = df["description_day0"].fillna("").tolist()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"{len(texts):,} dev descriptions | device = {device}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for short, hf_id in MODELS.items():
        t0 = time.time()
        revision = HfApi().model_info(hf_id).sha            # exact commit hash, then load THAT commit
        model = SentenceTransformer(hf_id, device=device, revision=revision)
        model.max_seq_length = MAX_SEQ_LEN
        emb = model.encode(texts, batch_size=BATCH, normalize_embeddings=True, convert_to_numpy=True,
                           show_progress_bar=True)
        np.save(OUT_DIR / f"{short}.npy", emb.astype(np.float16))
        df[["cve_id"]].to_parquet(OUT_DIR / f"{short}_ids.parquet", index=False)
        meta = {"model": hf_id, "revision": revision, "dim": int(emb.shape[1]), "n": int(emb.shape[0]), "device": device,
                "max_seq_length": MAX_SEQ_LEN, "normalized": True, "dtype": "float16", "input": "data/dev.parquet",
                "text_column": "description_day0", "sentence_transformers": sentence_transformers.__version__,
                "torch": torch.__version__, "seconds": round(time.time() - t0, 1)}
        (OUT_DIR / f"{short}_meta.json").write_text(json.dumps(meta, indent=1))
        print(f"{short}: {emb.shape} in {meta['seconds']}s -> {OUT_DIR.relative_to(ROOT)}/{short}.npy", flush=True)


# ------------------------------------------------------------------ Step 6c: fine-tuning (MiniLM)
FT_DIR = ROOT / "data" / "finetune"
FT_SETTINGS = {   # fixed in advance (max 4): learning rate x freeze lower half of the 6 transformer layers
    "lr1e-5_full": {"lr": 1e-5, "freeze_lower": False},
    "lr2e-5_full": {"lr": 2e-5, "freeze_lower": False},
    "lr1e-5_freeze": {"lr": 1e-5, "freeze_lower": True},
    "lr2e-5_freeze": {"lr": 2e-5, "freeze_lower": True},
}
FT = {"max_epochs": 3, "max_len": 128, "batch": 32, "weight_decay": 0.01, "dropout": 0.1, "neg_frac": 0.25,
      "warmup_frac": 0.1}
FT_GRID_BUDGET_MIN = None   # user decision: run all 4 settings (no time box on the grid)
FT_SUBSAMPLE_SEED = 12345   # fixed: identical training/stopping CVEs for all settings and training seeds


def _periods(df):
    """Same three periods as src/tabular_models.views: (name, training-period mask, evaluation mask)."""
    pub, tr = df["published"], df["split"].eq("train").to_numpy()
    return [("wf_2023", (pub.dt.year == 2022).to_numpy(), (pub.dt.year == 2023).to_numpy()),
            ("wf_2024h1", (pub.dt.year <= 2023).to_numpy(), (tr & (pub.dt.year == 2024)).to_numpy()),
            ("val", tr, df["split"].eq("val").to_numpy())]


def _finetune_run(df, setting, seed, device):
    import json
    import time

    import numpy as np
    import pandas as pd
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

    meta = json.loads((ROOT / "data" / "embeddings" / "minilm_meta.json").read_text())
    hf_id, rev = meta["model"], meta["revision"]
    tok = AutoTokenizer.from_pretrained(hf_id, revision=rev, cache_dir=os.environ["HF_HOME"])
    cfg = FT_SETTINGS[setting]
    y_all = df["label"].to_numpy().astype("float32")
    texts = df["description_day0"].fillna("").tolist()

    class Clf(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # frozen layers -> PyTorch picks an MPS attention kernel without dropout support; "eager" attention
            # computes the same thing and supports dropout. Full fine-tuning keeps the default (sdpa).
            extra = {"attn_implementation": "eager"} if cfg["freeze_lower"] else {}
            self.enc = AutoModel.from_pretrained(hf_id, revision=rev, cache_dir=os.environ["HF_HOME"], **extra)
            self.drop = torch.nn.Dropout(FT["dropout"])
            self.head = torch.nn.Linear(self.enc.config.hidden_size, 1)
            if cfg["freeze_lower"]:            # embeddings + lower half of the layers stay as pre-trained
                for p in self.enc.embeddings.parameters():
                    p.requires_grad = False
                n = len(self.enc.encoder.layer)
                for layer in self.enc.encoder.layer[: n // 2]:
                    for p in layer.parameters():
                        p.requires_grad = False

        def forward(self, ids, mask):
            h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
            pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)   # mean pooling (as MiniLM)
            return self.head(self.drop(pooled)).squeeze(-1)

    def subsample(mask, rng):          # all positives + NEG_FRAC of negatives
        idx = np.flatnonzero(mask)
        keep = (y_all[idx] == 1) | (rng.random(len(idx)) < FT["neg_frac"])
        return idx[keep]

    def batches(idx, shuffle, rng):
        order = rng.permutation(idx) if shuffle else idx
        for i in range(0, len(order), FT["batch"] if shuffle else 256):
            b = order[i:i + (FT["batch"] if shuffle else 256)]
            enc = tok([texts[j] for j in b], padding=True, truncation=True, max_length=FT["max_len"], return_tensors="pt")
            yield b, enc["input_ids"].to(device), enc["attention_mask"].to(device)

    def predict(model, idx):
        model.eval(); out = []
        with torch.no_grad():
            for _, ids, m in batches(idx, False, None):
                out.append(model(ids, m).float().cpu())
        return torch.cat(out).numpy()

    def train(idx_tr, n_epochs, idx_es=None):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = Clf().to(device)
        pos_w = torch.tensor(float((y_all[idx_tr] == 0).sum() / max((y_all[idx_tr] == 1).sum(), 1)),
                             dtype=torch.float32, device=device)   # MPS has no float64
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=FT["weight_decay"])
        steps = n_epochs * int(np.ceil(len(idx_tr) / FT["batch"]))
        sched = get_linear_schedule_with_warmup(opt, int(FT["warmup_frac"] * steps), steps)
        hist = []
        for ep in range(1, n_epochs + 1):
            model.train(); tot, n = 0.0, 0
            for b, ids, m in batches(idx_tr, True, rng):
                loss = loss_fn(model(ids, m), torch.tensor(y_all[b], device=device))
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                tot += loss.item() * len(b); n += len(b)
            row = {"epoch": ep, "train_loss": tot / n}
            if idx_es is not None:
                es_logits = torch.tensor(predict(model, idx_es))
                row["es_loss"] = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w.cpu())(es_logits, torch.tensor(y_all[idx_es])).item()
            hist.append(row)
            print(f"      epoch {ep}: {row}", flush=True)
        return model, hist

    FT_DIR.mkdir(parents=True, exist_ok=True)
    for name, fit_mask, ev_mask in _periods(df):
        t0 = time.time()
        last = df.loc[fit_mask, "published"].max().to_period("M")
        in_es = df["published"].dt.to_period("M").isin(pd.period_range(last - 5, last, freq="M")).to_numpy()
        sub = np.random.default_rng(FT_SUBSAMPLE_SEED)                  # same subsample for every setting/seed
        idx_search, idx_es = subsample(fit_mask & ~in_es, sub), subsample(fit_mask & in_es, sub)
        idx_full = np.union1d(idx_search, idx_es)                        # refit = search part + stopping part
        print(f"  [{setting} seed {seed} {name}] search {len(idx_search):,} / stop {len(idx_es):,} / refit {len(idx_full):,} / eval {ev_mask.sum():,}", flush=True)
        _, hist = train(idx_search, FT["max_epochs"], idx_es)
        best = int(min(hist, key=lambda r: r["es_loss"])["epoch"])
        model, refit_hist = train(idx_full, best)                       # same seed, full period, best epochs
        scores = predict(model, np.flatnonzero(ev_mask))
        pd.DataFrame({"cve_id": df.loc[ev_mask, "cve_id"].to_numpy(), "score": scores}).to_parquet(
            FT_DIR / f"{setting}_s{seed}_{name}.parquet", index=False)
        info = {"setting": setting, **cfg, **FT, "seed": seed, "subsample_seed": FT_SUBSAMPLE_SEED, "period": name,
                "n_search": len(idx_search), "n_stop": len(idx_es), "n_refit": len(idx_full),
                "best_epoch": best, "search_history": hist,
                "refit_history": refit_hist, "model": hf_id, "revision": rev, "device": device,
                "minutes": round((time.time() - t0) / 60, 1)}
        (FT_DIR / f"{setting}_s{seed}_{name}.json").write_text(json.dumps(info, indent=1))
        print(f"  [{setting} seed {seed} {name}] best epoch {best}, {info['minutes']} min", flush=True)
        del model
        if device == "mps":
            torch.mps.empty_cache()


def finetune(mode, setting=None, more=()):
    os.environ["HF_HUB_OFFLINE"] = "1"          # no network calls from this unsandboxed script
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import pandas as pd
    assert INPUT.name == "dev.parquet"                     # guard: this script never reads the test set
    df = pd.read_parquet(INPUT, columns=["cve_id", "published", "split", "label", "description_day0"])
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device != "mps":   # inside the sandbox MPS is invisible; never fall back to hours of CPU training
        raise SystemExit("MPS not available: run exactly `.venv/bin/python src/gpu_run.py finetune ...` "
                         "(no redirection), so the sandbox exemption applies")
    print(f"finetune {mode}: {len(df):,} dev CVEs | device = {device}", flush=True)
    if mode == "grid":
        import time
        t0 = time.time()
        for s in FT_SETTINGS:              # time box: no new setting after 60 min (leaves room for 2 more seeds)
            if FT_GRID_BUDGET_MIN is not None and (time.time() - t0) / 60 > FT_GRID_BUDGET_MIN:
                print(f"time box reached ({FT_GRID_BUDGET_MIN} min): skipping {s} and later settings", flush=True)
                break
            _finetune_run(df, s, 0, device)
    elif mode == "one":
        for s in (setting, *more):
            assert s in FT_SETTINGS, f"unknown setting {s}"
        for s in (setting, *more):
            _finetune_run(df, s, 0, device)
    elif mode == "seeds":                  # user decision (time budget): one extra seed -> 2 seeds in total
        assert setting in FT_SETTINGS, f"unknown setting {setting}"
        _finetune_run(df, setting, 1, device)


def embed_variant(name: str) -> None:
    import json
    import re
    import time

    import numpy as np
    import pandas as pd
    from sentence_transformers import SentenceTransformer

    assert re.fullmatch(r"[a-z0-9_]+", name), "variant name: lowercase letters, digits, _"
    os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
    vdir = OUT_DIR / "variants"
    df = pd.read_parquet(vdir / f"{name}_input.parquet", columns=["cve_id", "text"])
    dev_ids = set(pd.read_parquet(INPUT, columns=["cve_id"])["cve_id"])
    assert df["cve_id"].isin(dev_ids).all(), "embed-variant is for DEV CVEs only"
    meta0 = json.loads((OUT_DIR / "minilm_meta.json").read_text())
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    t0 = time.time()
    model = SentenceTransformer(meta0["model"], device=device, revision=meta0["revision"])
    model.max_seq_length = meta0["max_seq_length"]
    emb = model.encode(df["text"].fillna("").tolist(), batch_size=BATCH, normalize_embeddings=True,
                       convert_to_numpy=True)
    np.save(vdir / f"{name}_minilm.npy", emb.astype(np.float16))
    meta = {**meta0, "n": int(emb.shape[0]), "device": device, "input": f"variants/{name}_input.parquet",
            "seconds": round(time.time() - t0, 1)}
    (vdir / f"{name}_minilm_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{name}: {emb.shape} on {device} in {meta['seconds']}s (revision {meta0['revision'][:8]})", flush=True)


def embed_test() -> None:
    import json
    import time

    os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"   # before importing HF libraries
    import numpy as np
    import pandas as pd
    import sentence_transformers
    from sentence_transformers import SentenceTransformer

    sys.path.insert(0, str(ROOT / "src"))
    from description_leak import _phrase_names, drop_sentences_with

    if not torch.backends.mps.is_available():
        sys.exit("MPS not available - refusing to run (run exactly: .venv/bin/python src/gpu_run.py embed-test)")
    test_path = ROOT / "data" / "test_locked.parquet"
    df = pd.read_parquet(test_path, columns=["cve_id", "description_day0"])       # the ONLY columns read
    assert list(df.columns) == ["cve_id", "description_day0"]
    out = OUT_DIR / "test"
    out.mkdir(parents=True, exist_ok=True)
    meta0 = json.loads((OUT_DIR / "minilm_meta.json").read_text())                  # pinned dev revision
    t0 = time.time()
    model = SentenceTransformer(meta0["model"], device="mps", revision=meta0["revision"])
    model.max_seq_length = meta0["max_seq_length"]
    enc = lambda texts: model.encode(texts, batch_size=BATCH, normalize_embeddings=True, convert_to_numpy=True,
                                     show_progress_bar=True)
    texts = df["description_day0"].fillna("").tolist()
    print(f"{len(texts):,} test descriptions | device = mps | revision {meta0['revision'][:8]}", flush=True)
    emb = enc(texts)
    np.save(out / "minilm.npy", emb.astype(np.float16))
    df[["cve_id"]].to_parquet(out / "minilm_ids.parquet", index=False)
    meta = {**meta0, "n": int(emb.shape[0]), "device": "mps", "input": "data/test_locked.parquet",
            "columns_read": ["cve_id", "description_day0"], "offline": True,
            "sentence_transformers": sentence_transformers.__version__, "torch": torch.__version__,
            "seconds": round(time.time() - t0, 1)}
    (out / "minilm_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"test: {emb.shape} in {meta['seconds']}s -> data/embeddings/test/minilm.npy", flush=True)

    ph = df[[bool(_phrase_names(t)) for t in texts]].copy()
    ph["text"] = [drop_sentences_with(t, _phrase_names(t)) for t in ph["description_day0"]]
    assert not any(_phrase_names(t) for t in ph["text"])
    ph[["cve_id", "text"]].to_parquet(out / "phrase_removed_input.parquet", index=False)
    np.save(out / "phrase_removed_minilm.npy", enc(ph["text"].tolist()).astype(np.float16))
    print(f"phrase-removed variant: {len(ph)} test rows with a phrase -> data/embeddings/test/phrase_removed_*",
          flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "embed":
        embed()
    elif len(sys.argv) == 2 and sys.argv[1] == "embed-test":
        embed_test()
    elif len(sys.argv) > 2 and sys.argv[1] == "embed-variant":
        embed_variant(sys.argv[2])
    elif len(sys.argv) > 2 and sys.argv[1] == "finetune":
        finetune(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None, tuple(sys.argv[4:]))
    else:
        print(torch.backends.mps.is_available())
