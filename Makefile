PYTHON ?= python

.PHONY: preprocess build_rlset train ope xai distill serve test

preprocess:
	$(PYTHON) scripts/01_preprocess_log.py --config configs/config.yaml

build_rlset:
	$(PYTHON) scripts/02_encode_prefixes.py --config configs/config.yaml
	$(PYTHON) scripts/03_build_mdp_dataset.py --config configs/config.yaml

train:
	$(PYTHON) scripts/04_train_tdqn_offline.py --config configs/config.yaml

ope:
	$(PYTHON) scripts/05_run_ope_dr.py --config configs/config.yaml

xai:
	$(PYTHON) scripts/06_explain_policy.py --config configs/config.yaml

distill:
	$(PYTHON) scripts/08_distill_policy.py --config configs/config.yaml

serve:
	$(PYTHON) scripts/policy_server.py

test:
	pytest

# Regenerate every JSON the ICPM paper cites (docs/paper/MAPPING.md), assuming
# per-dataset D_offline.npz/splits/checkpoints exist. OPE for the 10 configs
# (primary + variant) plus the fidelity/evaluability/statistics artifacts.
PAPER_DATASETS = simbank simbank-ir3 bpi2012 bpi2017 bpi2017ct bpi2020-rfp \
	bpi2020-int-decl bpi2020-travel bpi2012-offertes sepsis

paper-tables:
	for ds in $(PAPER_DATASETS); do \
		$(PYTHON) scripts/05_run_ope_dr.py --dataset $$ds --behavior boa_logreg && \
		$(PYTHON) scripts/05_run_ope_dr.py --dataset $$ds --behavior tdqn_encoder; \
	done
	$(PYTHON) scripts/23_margin_drop_compare.py
	$(PYTHON) scripts/24_absgap_compare.py
	$(PYTHON) scripts/25_absgap_final.py
	$(PYTHON) scripts/26_fidelity_bpi2017ct.py
	$(PYTHON) scripts/26_fidelity_bpi2017ct.py simbank-ir3
	$(PYTHON) scripts/28_cross_level_tests.py
	$(PYTHON) scripts/32_evaluability.py
	$(PYTHON) scripts/ranking_separation.py
	$(PYTHON) scripts/29_selection_effect_bpi2012.py
	$(PYTHON) scripts/39_paper_stats.py
	$(PYTHON) scripts/34_audit_chain_versions.py --expect 2
