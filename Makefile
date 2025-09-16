# Makefile
# =========================================================
# RL Pong project: manage training, evaluation, and analysis
# =========================================================

# Environment variables
export PYTHONDONTWRITEBYTECODE=1        # Avoids writing .pyc files
export PYTORCH_ENABLE_MPS_FALLBACK=1    # Ensures MPS backend falls back gracefully

VENV = .venv
SEEDS = 42 84 126
ENV_ID = ALE/Pong-v5
FRAMES = 1500000

# -------------------------------
# Setup / Install / Cleanup
# -------------------------------
.PHONY: all setup install clean clean-run-data train-dqn-vanilla train-dqn-enhanced train-ppo train-all notebook rebuild tensorboard watch

all: install

# Create virtual environment + install dependencies
setup:
	@if [ ! -d "$(VENV)" ]; then \
		echo "🐍 Creating virtual environment..."; \
		python3 -m venv $(VENV); \
		echo "🐍 Installing certifi to handle SSL..."; \
		$(VENV)/bin/pip install certifi --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org; \
		CERT="$$( $(VENV)/bin/python -m certifi )"; \
		echo "🐍 Using cert bundle at: $$CERT"; \
		echo "🐍 Upgrading pip with custom certs..."; \
		$(VENV)/bin/pip install --upgrade pip --cert=$$CERT --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org; \
		echo "🐍 Installing all dependencies from requirements.txt with custom certs..."; \
		$(VENV)/bin/pip install -r requirements.txt --cert=$$CERT --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org; \
		echo "✅ Installation complete."; \
	else \
		echo "✅ Virtual environment already exists. Skipping installation."; \
	fi

install: setup

# Remove environment + all outputs
clean:
	@echo "🧹 Cleaning up entire project..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@rm -rf $(VENV) .pytest_cache .ipynb_checkpoints
	@echo "🧹 Deleting ALL generated output..."
	@rm -rf output
	@echo "✅ Full cleanup complete."

# Rebuild from scratch
rebuild: clean install
	@echo "✅ Environment has been successfully rebuilt."

# Remove outputs for one agent/seed combination
clean-run-data:
	@echo "🧹 Cleaning previous run data for $(agent)_$(seed)..."
	@rm -rf output/$(agent)_$(seed)

# -------------------------------
# Training targets
# -------------------------------
train-dqn-vanilla: install
	@echo "🏋️ Training Vanilla DQN..."
	@for seed in $(SEEDS); do \
		$(MAKE) clean-run-data agent=dqn_vanilla seed=$$seed; \
		echo "  -> Running with seed $$seed (artifacts in output/dqn_vanilla_$$seed)"; \
		$(VENV)/bin/python scripts/train.py --config configs/dqn_vanilla.yaml --seed $$seed --frames $(FRAMES); \
	done

train-dqn-enhanced: install
	@echo "🏋️ Training Enhanced (Double+Dueling) DQN..."
	@for seed in $(SEEDS); do \
		$(MAKE) clean-run-data agent=dqn_enhanced seed=$$seed; \
		echo "  -> Running with seed $$seed (artifacts in output/dqn_enhanced_$$seed)"; \
		$(VENV)/bin/python scripts/train.py --config configs/dqn_enhanced.yaml --seed $$seed --frames $(FRAMES); \
	done

train-ppo: install
	@echo "🏋️ Training PPO..."
	@for seed in $(SEEDS); do \
		$(MAKE) clean-run-data agent=ppo seed=$$seed; \
		echo "  -> Running with seed $$seed (artifacts in output/ppo_$$seed)"; \
		$(VENV)/bin/python scripts/train.py --config configs/ppo.yaml --seed $$seed --frames $(FRAMES); \
	done

# Run all experiments cleanly
train-all: clean install
	@echo "🔥 Running ALL experiments from a clean slate..."
	@$(MAKE) train-dqn-enhanced
	@$(MAKE) train-dqn-vanilla
	@$(MAKE) train-ppo

# -------------------------------
# Analysis / Visualization
# -------------------------------
notebook: install
	@echo "📓 Launching Jupyter Notebook..."
	@$(VENV)/bin/jupyter notebook notebooks/analysis.ipynb

tensorboard: install
	@echo "📊 Launching TensorBoard to monitor live runs..."
	@$(VENV)/bin/tensorboard --logdir output

# -------------------------------
# Watch a trained agent playing Pong
# -------------------------------
# Defaults: agent=ppo, seed=42, checkpoint auto-detected
# Usage:
#   make watch                 # uses ppo_42 and model_final.pt
#   make watch agent=ppo seed=84
#   make watch agent=dqn_enhanced seed=126 checkpoint=checkpoint.pt

agent ?= ppo
seed ?= 42
checkpoint ?=

watch: install
	@echo "👀 Watching $(agent)_$(seed) playing Pong..."
	@DIR="output/$(agent)_$(seed)/snapshots"; \
	if [ ! -d $$DIR ]; then \
		echo "❌ Directory not found: $$DIR"; exit 1; \
	fi; \
	if [ -z "$(checkpoint)" ]; then \
		if [ -f $$DIR/checkpoint.pt ]; then \
			CKPT="checkpoint.pt"; \
		elif [ -f $$DIR/model_final.pt ]; then \
			CKPT="model_final.pt"; \
		else \
			echo "❌ No checkpoint found in $$DIR"; exit 1; \
		fi; \
	else \
		CKPT="$(checkpoint)"; \
	fi; \
	$(VENV)/bin/python scripts/evaluate.py \
		--config configs/$(agent).yaml \
		--model-path $$DIR/$$CKPT \
		--env-id ALE/Pong-v5 \
		--episodes 5