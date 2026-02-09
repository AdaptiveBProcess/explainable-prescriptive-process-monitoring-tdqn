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


