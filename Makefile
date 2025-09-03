# Makefile
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

VENV = .venv
SEEDS = 42 84 126
ENV_ID = ALE/Pong-v5
FRAMES = 3000000

.PHONY: all setup install clean clean-run-data train-dqn-vanilla train-dqn-enhanced train-ppo train-all notebook rebuild tensorboard

all: install

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

clean:
	@echo "🧹 Cleaning up entire project..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@rm -rf $(VENV) .pytest_cache .ipynb_checkpoints
	@echo "🧹 Deleting ALL generated output..."
	@rm -rf output
	@echo "✅ Full cleanup complete."

rebuild: clean install
	@echo "✅ Environment has been successfully rebuilt."

# Clean data for a SPECIFIC agent and seed combination
clean-run-data:
	@echo "🧹 Cleaning previous run data for $(agent)_$(seed)..."
	@rm -rf output/$(agent)_$(seed)

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

train-all: clean install
	@echo "🔥 Running ALL experiments from a clean slate..."
	@$(MAKE) train-dqn-enhanced
	@$(MAKE) train-dqn-vanilla
	@$(MAKE) train-ppo

notebook: install
	@echo "📓 Launching Jupyter Notebook..."
	@$(VENV)/bin/jupyter notebook notebooks/analysis.ipynb

tensorboard: install
	@echo "📊 Launching TensorBoard to monitor live runs..."
	@$(VENV)/bin/tensorboard --logdir output