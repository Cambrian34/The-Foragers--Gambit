# The Foragers' Gambit

Neuroevolution of Emergent Foraging and Communication Strategies in a Multi-Agent System

## Overview

This repository contains code, data and notes for experiments on evolving multi-agent foraging and communication strategies using neuroevolution. Agents ("creatures") are controlled by neural networks whose weights are evolved across generations. The experiments explore how emergent behaviors such as cooperative foraging, communication (signalling), and simple memory/fleeing strategies arise when varying network architectures, mutation rates, and environmental parameters.



## Key ideas

- Agents perceive the environment (nearby apples, other agents, home/base) and output movement and signalling actions.
- Genomes encode the neural network weights. Evolutionary operators (mutation, crossover) are applied to populations to improve task performance.
- Fitness measures include apples collected, apples deposited, group survival, and emergent cooperative metrics.

## Repository structure
- Phase 1: Basic foraging and survival behaviors
- `Neuroevolution Swarm Behavior Research Project-2.pdf` — the full project report describing methods and results.(Currently in draft and is not included in the repo)

## Assumptions and notes

1. Assumes a Python 3 environment (3.8+ recommended).
2. The project uses common scientific Python libraries (numpy, matplotlib, pandas). The visual simulation may require `pygame` (optional).
3. Some scripts expect relative paths and will read/write files in the repository root. Run scripts from the repo root to avoid path issues.

If you need a more precise environment, see `requirements.txt` included here as a starting point.

## Quick start

Install dependencies into a virtual environment (recommended):

```bash
# macOS / zsh example
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Recommended parameters 

- Population: 50
- Generations: 100–1000 (reduce for quick tests)
- Mutation rate: 0.01–0.1 (several experiments vary this parameter. Use multiple runs to see effects)


## Outputs and interpreting results

- CSV rows typically record per-generation metrics: best fitness, average fitness, apples collected, apples deposited, creatures remaining.
- Use `plots/` to view aggregated comparisons across conditions (baseline vs mutated settings, architecture changes, etc.).

## Troubleshooting

- If visual scripts fail, ensure `pygame` is installed and your display is available.
- If you get path errors, run scripts from the repo root and update any hard-coded paths in the top of the script.

## Extensions and next steps

- (Projected)Containerize the environment (Docker) for fully reproducible experiments on other machines or clusters.

## Credits & License

Author: Alistair Chambers 

This repository contains research code. Please contact the author before commercial reuse. If you reuse or adapt results, cite the research notes in `Neuroevolution Swarm Behavior Research Project-2.pdf`.

## Contact

If you want help reproducing experiments or extending the code, open an issue or contact the repository owner.
Email: alistair.chambers@my.utsa.edu