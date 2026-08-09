# Generality study: PL-xPsPM on external PPO agents (paper Sec. 4.5)

Instantiates the paper's dual-level explanation method (risk = IG on the
critic V(s); margin = IG on the actor's logit margin) on two *foreign*
prescriptive-process-monitoring agents trained with stable-baselines3 PPO,
to show the method is not an artifact of our own TDQN.

Every number in the paper's Generality subsection comes from the two JSONs in
`../../artifacts/generality/`:

| Paper claim | JSON field |
|---|---|
| Primary agent never intervenes (margin not evaluable) | `rl-prescriptive-monitoring.json: intervene_rate, n_margin_states, margin_evaluable_sample` |
| Risk test passes on the primary agent (gap/SE at k=1,2) | `rl-prescriptive-monitoring.json: value_test.phi_v.{1,2}.gap / gap_se` |
| >80% of risk mass on `available_resources` | `rl-prescriptive-monitoring.json: phi_v_mean_abs` (share of the 4th entry) |
| when-to-treat margin gap in SEs, flips guided vs random | `when-to-treat.json: margin_test.phi_dq.1.{gap,gap_se,flip_guided,flip_random}` |
| Margin drain blocked by Def. 4(iii) | `when-to-treat.json: margin_above_null` (mean abs margin vs null) |
| Critic fails its own value test | `when-to-treat.json: value_test.phi_v` (negative/ns gaps) |
| φ^ΔQ mass concentrated on `lower_TE` (CATE link) | `when-to-treat.json: phi_dq_mean_abs` (share of 2nd entry) |

## Layout

- `dual_level.py` — the method: differentiable `CriticHead`/`MarginHead` over an
  SB3 `ActorCriticPolicy`, feature-space Integrated Gradients with a
  background-mean reference, deletion tests (guided/random/anti |displacement|,
  paired SEs, sign flips), and `dual_level_study` orchestrating both levels.
- `run_dual_level_ppo.py` — the experiment. Regenerates the two JSONs
  end-to-end (deterministic, seed 123).
- `test_dual_level.py` — smoke tests on an untrained PPO (IG completeness,
  margin-head consistency, deletion shapes). Outside the main suite's
  testpaths on purpose: the main `.venv` has no stable-baselines3.
- `models/` — the two PPO checkpoints (committed; ~140 KB each).
- `foreign/train_ppo_fast_*.py` — the training scripts that produced them.
  They reimplement each third-party repo's env directly from its CSV
  (bypassing TensorFlow-era code), faithful to the original state and reward
  design. Primary agent: 300k steps; when-to-treat: 200k steps; seed 42.

## Third-party inputs (not committed: 478 MB + 186 MB)

The state CSVs are the third-party repos' own preprocessed BPIC 2017 data:

- `RL-prescriptive-monitoring/rl/data/ready_to_use_adaptive_bpic2017.csv`
  from https://github.com/mshoush/RL-prescriptive-monitoring
  (Shoush & Dumas, "Prescriptive Process Monitoring Under Resource
  Constraints: A Reinforcement Learning Approach").
- `WhenToTreat/RL/data/results_adaptive_counterfacs_bpic2017.csv`
  from https://github.com/zahradbozorgi/WhenToTreat — produced by its
  counterfactual treatment-effect-bound estimation step.

Both are regenerable from the public BPIC 2017 event log with those repos'
preprocessing. Place both checkouts under one directory and pass it as
`--lib` (or `XPPM_PPO_LIB`).

## Regenerate

```bash
python -m venv .venv-ppo && . .venv-ppo/bin/activate
pip install -r experiments/generality_ppo/requirements.txt
python scripts/37_ppo_transplant.py --lib <dir-with-both-checkouts>
pytest experiments/generality_ppo/test_dual_level.py -v
```
