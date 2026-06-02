# VoiceLens — common tasks.
# Usage: `make <target>`. On Windows, run via Git Bash / WSL, or run the
# underlying commands directly (see README Quick start).

ifeq ($(OS),Windows_NT)
  VENV_PY := .venv/Scripts/python.exe
else
  VENV_PY := .venv/bin/python
endif

.PHONY: install backend frontend dev finetune eval test clean help

help:
	@echo "install   - create .venv, install backend + frontend deps"
	@echo "backend   - run the FastAPI server (http://localhost:8000)"
	@echo "frontend  - run the Vite dev server (http://localhost:5173)"
	@echo "finetune  - prepare the dataset and run QLoRA training (GPU/Colab)"
	@echo "eval      - WER comparison: base vs fine-tuned (GPU/Colab)"
	@echo "test      - run the Python unit tests"

install:
	python -m venv .venv
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install -r requirements.txt
	cd frontend && npm install

backend:
	$(VENV_PY) -m uvicorn backend.main:app --reload

frontend:
	cd frontend && npm run dev

# Run backend + frontend together (needs a shell with job control).
dev:
	$(VENV_PY) -m uvicorn backend.main:app --reload & cd frontend && npm run dev

finetune:
	$(VENV_PY) finetune/prepare_dataset.py
	$(VENV_PY) finetune/train.py

eval:
	$(VENV_PY) finetune/evaluate.py

test:
	$(VENV_PY) -m unittest discover -s tests -t .

clean:
	rm -rf finetune/data finetune/output/adapter backend/voicelens.db frontend/dist
