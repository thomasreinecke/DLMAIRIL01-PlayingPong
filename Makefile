# Makefile
export PYTHONDONTWRITEBYTECODE=1
# Ensure PyTorch gracefully falls back for unsupported MPS ops
export PYTORCH_ENABLE_MPS_FALLBACK=1

VENV = .venv
SEEDS = 42 84 126
ENV_ID = ALE/Pong-v5
FRAMES = 5000000
HEARTBEAT_SECS ?= 10

.PHONY: all setup install deps clean train-dqn-vanilla train-dqn-enhanced train-ppo train-all notebook tensorboard

all: install

# Create venv and install base requirements (once)
setup:
	@if [ ! -d "$(VENV)" ]; then \
		echo "🐍 Creating virtual environment."; \
		python3 -m venv $(VENV); \
		echo "🐍 Installing certifi."; \
		$(VENV)/bin/pip install certifi --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org; \
		echo "🐍 Capturing certifi certificate bundle path and upgrading pip/installing requirements."; \
		CERT="$$( $(VENV)/bin/python -m certifi )"; \
		echo "Certifi installed at: $$CERT"; \
		echo "🐍 Upgrading pip."; \
		$(VENV)/bin/pip install --upgrade pip --cert=$$CERT --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org; \
		echo "🐍 Installing remaining requirements."; \
		$(VENV)/bin/pip install -r requirements.txt --cert=$$CERT --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org; \
		echo "✅ Installation complete."; \
	else \
		echo "✅ Virtual environment already exists. Skipping installation."; \
	fi

# Always ensure runtime deps that Gymnasium's Atari wrappers need (e.g., OpenCV)
deps:
	@echo "🔧 Ensuring runtime dependencies (OpenCV for AtariPreprocessing)..."
	@CERT="$$( $(VENV)/bin/python -m certifi )"; \
	$(VENV)/bin/pip install opencv-python --cert=$$CERT --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org >/dev/null

install: setup deps

# Removes the virtual environment and pycache folders
clean:
	@echo "🧹 Cleaning up..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@rm -rf $(VENV) .pytest_cache .ipynb_checkpoints
	@echo "🧹 Deleting logs and data..."
	@rm -rf runs/* data/*

# Train the vanilla DQN agent across all seeds
train-dqn-vanilla: install
	@echo "🏋️ Training Vanilla DQN..."
	@for seed in $(SEEDS); do \
		echo "  -> Running with seed $$seed"; \
		$(VENV)/bin/python scripts/train.py --config configs/dqn_vanilla.yaml --seed $$seed --frames $(FRAMES) --env-id $(ENV_ID) --heartbeat-secs $(HEARTBEAT_SECS); \
	done

# Train the enhanced DQN agent across all seeds
train-dqn-enhanced: install
	@echo "🏋️ Training Enhanced (Double+Dueling) DQN..."
	@for seed in $(SEEDS); do \
		echo "  -> Running with seed $$seed"; \
		$(VENV)/bin/python scripts/train.py --config configs/dqn_enhanced.yaml --seed $$seed --frames $(FRAMES) --env-id $(ENV_ID) --heartbeat-secs $(HEARTBEAT_SECS); \
	done

# Train the PPO agent across all seeds
train-ppo: install
	@echo "🏋️ Training PPO..."
	@for seed in $(SEEDS); do \
		echo "  -> Running with seed $$seed"; \
		$(VENV)/bin/python scripts/train.py --config configs/ppo.yaml --seed $$seed --frames $(FRAMES) --env-id $(ENV_ID) --heartbeat-secs $(HEARTBEAT_SECS); \
	done

# Convenience: run all experiments (order chosen to reuse caches/wheels etc.)
train-all: install
	@$(MAKE) train-dqn-enhanced
	@$(MAKE) train-dqn-vanilla
	@$(MAKE) train-ppo

# Launch Jupyter notebook for analysis
notebook: install
	@echo "📓 Launching Jupyter Notebook..."
	@$(VENV)/bin/jupyter notebook notebooks/analysis.ipynb

# Launch TensorBoard to view logs
tensorboard: install
	@echo "📊 Launching TensorBoard..."
	@$(VENV)/bin/tensorboard --logdir runs
