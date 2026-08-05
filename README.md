# Probabilistic Transformer Experiments

Эксперименты с вероятностными трансформерами.

## Структура

```
.
├── src/          # исходный код
├── experiments/  # конфиги и скрипты запусков
├── notebooks/    # анализ и визуализация
└── data/         # данные (не версионируются)
```

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Запуск

```bash
python -m src.train --config experiments/baseline.yaml
```
