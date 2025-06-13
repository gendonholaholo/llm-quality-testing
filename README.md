# llm-quality-testing

[![PyPI version](https://img.shields.io/pypi/v/llm-quality-testing.svg?style=flat)](https://pypi.org/project/llm-quality-testing/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Purpose
A CLI tool to compare the quality of multiple LLMs (Hugging Face) on the same dataset using perplexity, accuracy, and BLEU metrics. Results are displayed in the terminal (table) and can be saved to file (CSV/JSON) for documentation and further analysis.

## Installation

### From Local Repository
```bash
pip install .
```

### From PyPI (after publishing)
```bash
pip install llm-quality-testing
```

## YAML Configuration Structure
Example file: `configs/default_config.yaml`
```yaml
models:
  - gpt2
  - facebook/bart-large-cnn
dataset: data/sample_data.json
output_csv: results/leaderboard.csv
output_json: results/leaderboard.json
```
- `models`: List of Hugging Face model names to compare.
- `dataset`: Path to test data file (JSON, list of {"text", "label"}).
- `output_csv`: Output leaderboard file path (CSV).
- `output_json`: Output leaderboard file path (JSON).

## Usage

### Run with YAML config
```bash
llm-tester evaluate configs/default_config.yaml
```

### Override config values from CLI
```bash
llm-tester evaluate configs/default_config.yaml \
  --model-name gpt2 --model-name facebook/bart-large-cnn \
  --dataset-path data/sample_data.json \
  --output-csv results/leaderboard.csv \
  --output-json results/leaderboard.json \
  --metrics perplexity --metrics accuracy
```

### Example Output in Terminal
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                Comparative Model Leaderboard     ┃
┡━━━━━━━━━━━━━━━┯━━━━━━━━━━━━┯━━━━━━━━━━┯━━━━━━━━━━┩
│ model_name    │ perplexity │ accuracy │ bleu     │
├───────────────┼────────────┼──────────┼──────────┤
│ gpt2          │ 34.1234    │ 0.8125   │ 0.5123   │
│ bart-large-cnn│ 28.5678    │ 0.8450   │ 0.6012   │
└───────────────┴────────────┴──────────┴──────────┘

Leaderboard saved to: results/leaderboard.csv and results/leaderboard.json
```

## Output Files
- `results/leaderboard.csv`: Comparison results in CSV format.
- `results/leaderboard.json`: Same results in JSON format.

## Demo

[Demo GIF akan ditambahkan di sini]

## Contributing

Contributions are welcome! To contribute:
- Fork this repository and create a new branch for your feature or bugfix.
- Make your changes and add tests as needed.
- Submit a pull request describing your changes.

## Testing
Run all tests with:
```bash
pytest
```

## Project Structure
```
llm-quality-testing/
├── llm_eval/          # Core evaluation code
├── scripts/           # CLI scripts
├── tests/             # Unit & integration tests
├── configs/           # YAML config files
├── results/           # Leaderboard output
├── pyproject.toml
└── README.md
```

## Notes
- For large models, ensure sufficient resources (RAM/GPU).
- To add models, simply edit the YAML config file.
- The structure is ready for further extension (custom loader, parallelization, API, etc).
