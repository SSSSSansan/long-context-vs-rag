PY ?= python
EXP ?= level1

.PHONY: setup raw corpus run report estimate smoke test all clean

setup:
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && pip freeze > requirements.lock.txt

estimate:            ## смета до запуска
	$(PY) -m scripts.estimate_cost

smoke:               ## проверка клиента (локальный backend, бесплатно)
	LLM_BACKEND=local $(PY) -m scripts.smoke_llm

raw:                 ## синтетический фоновый корпус -> corpus/raw
	$(PY) -m src.gen_raw_corpus

corpus:              ## день 1-2: сборка 8k/30k/100k + факты + gold
	$(PY) -m src.build_corpus

run:                 ## make run EXP=level1
	$(PY) -m src.eval.run --exp configs/experiments/$(EXP)*.yaml

report:
	$(PY) -m src.eval.aggregate

test:
	pytest -q

all: corpus run report

clean:
	rm -rf runs/cache runs/budget.json results/*.csv results/charts/*
