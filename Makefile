.PHONY: help venv install data train promote up serve down logs clean

help:
	@echo "make venv      - create .venv"
	@echo "make install   - install deps (editable)"
	@echo "make data      - generate dataset + manifest"
	@echo "make train     - train + log to MLflow + register model"
	@echo "make promote   - promote model to Production"
	@echo "make up        - start MLflow"
	@echo "make serve     - start MLflow + inference"
	@echo "make down      - stop services"
	@echo "make logs      - follow docker logs"
	@echo "make clean     - remove local outputs"

venv:
	python -m venv .venv

install:
	. .venv/bin/activate && pip install -U pip && pip install -e ".[dev]"

data:
	. .venv/bin/activate && python -m src.data.make_dataset

train:
	. .venv/bin/activate && python -m src.training.train

promote:
	. .venv/bin/activate && python -m src.registry.promote --min-acc 0.80

up:
	docker compose up -d

serve:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f

clean:
	rm -rf artifacts
