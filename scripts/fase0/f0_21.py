"""Baseline attribution methods (gradient saliency, attention rollout) for the
fidelity comparison of the paper (plan-accion Fase 1, T1.1).

For each dataset, reuses the SAME selected transitions and target values as the
existing IG artifacts (risk_explanations.json / deltaQ_explanations.json) and
only replaces the token rankings (top/bottom tokens and drivers) with the ones
produced by each baseline method. Output JSONs keep the pipeline schema, so
scripts/07_fidelity_tests.py and the margin-drop apparatus run on them unchanged.

Standalone: does NOT modify pipeline artifacts; writes to artifacts/xai/baselines/.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from xppm.rl.train_tdqn import load_dataset_with_splits  # noqa: E402
from xppm.xai.fidelity_tests import _load_q_network  # noqa: E402

DATASETS = {
    "simbank": {
        "npz": REPO / "data/simbank/processed/D_offline.npz",
        "splits": REPO / "data/simbank/processed/splits.json",
        "xai": REPO / "artifacts/xai",
        "ope": REPO / "artifacts/ope/ope_dr.json",
    },
    "bpi2017": {
        "npz": REPO / "data/bpi2017/processed/D_offline.npz",
        "splits": REPO / "data/bpi2017/processed/splits.json",
        "xai": REPO / "artifacts/xai/bpi2017",
        "ope": REPO / "artifacts/ope/bpi2017/ope_dr.json",
    },
    "bpi2020-rfp": {
        "npz": REPO / "data/bpi2020-rfp/processed/D_offline.npz",
        "splits": REPO / "data/bpi2020-rfp/processed/splits.json",
        "xai": REPO / "artifacts/xai/bpi2020-rfp",
        "ope": REPO / "artifacts/ope/bpi2020-rfp/ope_dr.json",
    },
    "bpi2012": {
        "npz": REPO / "data/bpi2012/processed/D_offline.npz",
        "splits": REPO / "data/bpi2012/processed/splits.json",
        "xai": REPO / "artifacts/xai/bpi2012",
        "ope": REPO / "artifacts/ope/bpi2012/ope_dr.json",
    },
    "bpi2017ct": {
        "npz": REPO / "data/bpi2017ct/processed/D_offline.npz",
        "splits": REPO / "data/bpi2017ct/processed/splits.json",
        "xai": REPO / "artifacts/xai/bpi2017ct",
        "ope": REPO / "artifacts/ope/bpi2017ct/ope_dr.json",
    },
    "simbank-ir3": {
        "npz": REPO / "data/simbank-ir3/processed/D_offline.npz",
        "splits": REPO / "data/simbank-ir3/processed/splits.json",
        "xai": REPO / "artifacts/xai/simbank-ir3",
        "ope": REPO / "artifacts/ope/simbank-ir3/ope_dr.json",
    },
    "bpi2012-offertes": {
        "npz": REPO / "data/bpi2012-offertes/processed/D_offline.npz",
        "splits": REPO / "data/bpi2012-offertes/processed/splits.json",
        "xai": REPO / "artifacts/xai/bpi2012-offertes",
        "ope": REPO / "artifacts/ope/bpi2012-offertes/ope_dr.json",
    },
    "sepsis": {
        "npz": REPO / "data/sepsis/processed/D_offline.npz",
        "splits": REPO / "data/sepsis/processed/splits.json",
        "xai": REPO / "artifacts/xai/sepsis",
        "ope": REPO / "artifacts/ope/sepsis/ope_dr.json",
    },
}

CONFIG = {"training": {"transformer": {}}}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BATCH = 256  # chunk size; selections can reach thousands of items


def _batched(fn, s, sm, *rest) -> np.ndarray:
    outs = []
    for i in range(0, len(s), BATCH):
        outs.append(fn(s[i : i + BATCH], sm[i : i + BATCH], *(r[i : i + BATCH] for r in rest)))
    return np.concatenate(outs, axis=0)


def saliency_importance(q_net, s, sm, a_star, a_contrast=None) -> np.ndarray:
    """|d target / d embedding| L1-aggregated per position.

    target = Q(s, a_star) if a_contrast is None else Q(s,a_star) - Q(s,a_contrast),
    with the action(s) held fixed (same convention as the IG pipeline).
    Delegates the forward to q_net.encode so positional embeddings, the
    attention padding mask and the pooling rule match the deployed encoder
    version instead of replaying v1 semantics by hand.
    """
    if len(s) > BATCH:
        if a_contrast is None:
            return _batched(
                lambda s_, sm_, a_: saliency_importance(q_net, s_, sm_, a_), s, sm, a_star
            )
        return _batched(
            lambda s_, sm_, a_, c_: saliency_importance(q_net, s_, sm_, a_, c_),
            s,
            sm,
            a_star,
            a_contrast,
        )
    s_t = torch.from_numpy(s).long().to(DEVICE)
    sm_t = torch.from_numpy(sm).float().to(DEVICE)
    a_star_t = torch.from_numpy(a_star).long().to(DEVICE)

    s_clamped = torch.clamp(s_t, min=0, max=q_net.vocab_size - 1)
    emb = q_net.embedding(s_clamped).detach().requires_grad_(True)
    q = q_net.encode(emb, sm_t)

    target = q.gather(1, a_star_t.unsqueeze(1)).squeeze(1)
    if a_contrast is not None:
        a_con_t = torch.from_numpy(a_contrast).long().to(DEVICE)
        target = target - q.gather(1, a_con_t.unsqueeze(1)).squeeze(1)
    target.sum().backward()

    imp = emb.grad.detach().abs().sum(dim=2)  # (batch, max_len) L1 over d_model
    imp = imp * sm_t  # zero out PAD
    return imp.cpu().numpy()


def attention_rollout_importance(q_net, s, sm) -> np.ndarray:
    """Attention-rollout importance of each input position for the pooled state.

    Manually replays the (eval-mode, norm_first=False) TransformerEncoderLayer
    forward to capture head-averaged attention, verifies the replay against
    q_net.encoder, then rolls out A_hat = 0.5*A + 0.5*I across layers and reads
    the row of the pooled (last non-PAD) position. Target-agnostic by design.
    """
    if len(s) > BATCH:
        return _batched(lambda s_, sm_: attention_rollout_importance(q_net, s_, sm_), s, sm)
    s_t = torch.from_numpy(s).long().to(DEVICE)
    sm_t = torch.from_numpy(sm).float().to(DEVICE)
    with torch.no_grad():
        s_clamped = torch.clamp(s_t, min=0, max=q_net.vocab_size - 1)
        x = q_net.embedding(s_clamped)
        if q_net.pos_embedding is not None:
            pos = torch.arange(x.size(1), device=DEVICE).unsqueeze(0).expand(x.size(0), -1)
            x = x + q_net.pos_embedding(pos)
        key_padding = None
        if q_net.encoder_version >= 2:
            key_padding = sm_t <= 0
            key_padding = key_padding & ~key_padding.all(dim=1, keepdim=True)
        h = q_net.encoder.input_proj(x)
        attn_mats = []
        for layer in q_net.encoder.encoder.layers:
            attn_out, attn_w = layer.self_attn(
                h,
                h,
                h,
                key_padding_mask=key_padding,
                need_weights=True,
                average_attn_weights=True,
            )
            attn_mats.append(attn_w)  # (batch, L, L)
            h = layer.norm1(h + layer.dropout1(attn_out))
            ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(h))))
            h = layer.norm2(h + layer.dropout2(ff))
        ref = q_net.encoder(x, key_padding_mask=(sm_t <= 0) if key_padding is not None else None)
        replay_err = (h - ref).abs().max().item()
        if replay_err > 1e-3:
            raise RuntimeError(f"Manual encoder replay mismatch: {replay_err}")

        eye = torch.eye(h.size(1), device=DEVICE).unsqueeze(0)
        rollout = None
        for a_mat in attn_mats:
            a_hat = 0.5 * a_mat + 0.5 * eye
            a_hat = a_hat / a_hat.sum(dim=-1, keepdim=True)
            rollout = a_hat if rollout is None else torch.bmm(a_hat, rollout)

        idx = q_net._pool_index(sm_t)
        batch_idx = torch.arange(h.size(0), device=DEVICE)
        imp = rollout[batch_idx, idx]  # (batch, L): contribution of each pos
        imp = imp * sm_t
    return imp.cpu().numpy()


def replace_rankings(items, importance, s, sm, top_key, bottom_key, top_k=10):
    """Return items with top/bottom rankings replaced by `importance` order."""
    out = []
    for j, it in enumerate(items):
        imp = importance[j]
        top_idx = np.argsort(imp)[::-1][:top_k]
        nonpad = np.where(sm[j] > 0)[0]
        bottom_idx = nonpad[np.argsort(imp[nonpad])[:top_k]] if len(nonpad) else []
        new = dict(it)
        old_names = {t["position"]: t["token_name"] for t in it.get(top_key, [])}
        old_names.update({t["position"]: t["token_name"] for t in it.get(bottom_key, [])})

        def entry(idx):
            return {
                "position": int(idx),
                "token_id": int(s[j, idx]),
                "token_name": old_names.get(int(idx), f"tok_{int(s[j, idx])}"),
                "importance": float(imp[idx]),
            }

        new[top_key] = [entry(i) for i in top_idx if imp[i] > 0]
        new[bottom_key] = [entry(i) for i in bottom_idx]
        out.append(new)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--methods", nargs="+", default=["saliency", "attention"])
    ap.add_argument("--out-root", default=str(REPO / "artifacts/xai/baselines"))
    args = ap.parse_args()

    for name in args.datasets:
        p = DATASETS[name]
        ope = json.load(open(p["ope"]))
        ckpt = REPO / ope["metadata"]["ckpt_path"]
        vocab = REPO / ope["metadata"]["vocab_path"]
        print(f"== {name}: ckpt={ckpt.parent.name}")
        q_net = _load_q_network(ckpt, p["npz"], vocab, CONFIG, DEVICE)
        test = load_dataset_with_splits(str(p["npz"]), str(p["splits"]), "test")

        risk = json.load(open(p["xai"] / "risk_explanations.json"))
        dq = json.load(open(p["xai"] / "deltaQ_explanations.json"))
        r_items, d_items = risk["items"], dq["items"]

        idx_r = np.array([it["transition_idx"] for it in r_items])
        idx_d = np.array([it["transition_idx"] for it in d_items])
        assert (idx_r == idx_d).all(), "risk/deltaQ item order mismatch"
        s_sel, sm_sel = test["s"][idx_r], test["s_mask"][idx_r]
        a_star_r = np.array([it["a_star"] for it in r_items])
        a_star_d = np.array([it["a_star"] for it in d_items])
        a_con_d = np.array([it["a_contrast"] for it in d_items])

        per_method = {}
        if "saliency" in args.methods:
            per_method["saliency"] = {
                "risk": saliency_importance(q_net, s_sel, sm_sel, a_star_r),
                "dq": saliency_importance(q_net, s_sel, sm_sel, a_star_d, a_con_d),
            }
        if "attention" in args.methods:
            att = attention_rollout_importance(q_net, s_sel, sm_sel)
            per_method["attention"] = {"risk": att, "dq": att}

        for method, imps in per_method.items():
            out_dir = Path(args.out_root) / method / name
            out_dir.mkdir(parents=True, exist_ok=True)
            meta_r = {**risk["metadata"], "attribution_method": method}
            meta_d = {**dq["metadata"], "attribution_method": method}
            json.dump(
                {
                    "metadata": meta_r,
                    "items": replace_rankings(
                        r_items, imps["risk"], s_sel, sm_sel, "top_tokens", "bottom_tokens"
                    ),
                },
                open(out_dir / "risk_explanations.json", "w"),
                indent=1,
            )
            json.dump(
                {
                    "metadata": meta_d,
                    "items": replace_rankings(
                        d_items, imps["dq"], s_sel, sm_sel, "top_drivers", "bottom_drivers"
                    ),
                },
                open(out_dir / "deltaQ_explanations.json", "w"),
                indent=1,
            )
            for aux in ("explanations_selection.json", "policy_summary.json"):
                if (p["xai"] / aux).exists():
                    shutil.copy(p["xai"] / aux, out_dir / aux)
            print(f"   {method} -> {out_dir}")


if __name__ == "__main__":
    main()
