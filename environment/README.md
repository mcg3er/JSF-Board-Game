# PyTAG: a Reinforcement Learning interface for the [Tabletop Games Framework](http://www.tabletopgames.ai/)

[![license](https://img.shields.io/github/license/martinballa/PyTAG)](LICENSE)
![top-language](https://img.shields.io/github/languages/top/martinballa/PyTAG)
![code-size](https://img.shields.io/github/languages/code-size/martinballa/PyTAG)
[![twitter](https://img.shields.io/twitter/follow/gameai_qmul?style=social)](https://twitter.com/intent/follow?screen_name=gameai_qmul)
[![](https://img.shields.io/github/stars/martinballa/PyTAG.svg?label=Stars&style=social)](https://github.com/GAIGResearch/TabletopGames)

PyTAG allows interaction with the TAG framework from Python. This repository contains all the python code required to
run Reinforcement Learning agents.
The aim of PyTAG is to provide a Reinforcement Learning API for the TAG framework, but it is not limited to RL as using
the python-java bridge all public functions and variables are accessible from python.
If you want to learn more about TAG, please visit the [website](http://tabletopgames.ai).

You may try [this](https://colab.research.google.com/drive/1WMVu9bFkxvwK7evD1sIkxcsrlhdRoY9d?usp=sharing) google colab
notebook to try out PyTAG before installing it on your own machine.

## Setting up

For Pandemic on Windows, use the [single-command setup](#pandemic-on-windows) below instead of downloading the generic TAG jar.

TAG requires Java with minimum version 21. We recommend installing pytag in a new virtual environment. To 
install
pytag you may follow the steps below.

- 1, Clone this repository.
- 2, Install PyTAG as a python package ```pip install -e .```
- 3, Run ```python jar_setup.py``` to download the latest `TAG.jar` or see "Getting the TAG jar file" below for manual options.
- 4, (optional) install pytag with the additional dependencies to run the baselines ```pip install -e .[examples]```
- 5, (optional) you may test your installation by running the examples in ```examples/``` for instance
  ```pt-action-masking.py```.

### Getting the TAG jar file
PyTAG requires a single `TAG.jar` file placed in the `pytag/jars/` folder. Running `jar_setup.py` will download it automatically (no extra dependencies required):
```bash
python jar_setup.py
```
Or download `TAG.jar` manually from the [TAG releases page](https://github.com/GAIGResearch/TabletopGames/releases) and place it in `pytag/jars/`.

To build `TAG.jar` from source, see the [TAG wiki](https://tabletopgames.ai/wiki/maven): run `mvn package` in the TAG repository and copy `target/TAG-pytag.jar` to `pytag/jars/TAG.jar` (the slim PyTAG-specific build - not `target/TAG.jar`, which bundles unrelated ML/analytics dependencies PyTAG doesn't need).

#### Using a custom jar without touching `pytag/jars/`

If you're iterating on a local TAG checkout (e.g. testing a game or agent change before it's released),
you don't need to copy your build into `pytag/jars/TAG.jar` each time. Point PyTAG at it directly with the
`PYTAG_JAR_PATH` environment variable:

```bash
export PYTAG_JAR_PATH=/path/to/TabletopGames/target/TAG-pytag.jar
python examples/pt-action-masking.py
```

This takes priority over `pytag/jars/TAG.jar` for both `PyTAG`/`MultiAgentPyTAG`/`SelfPlayPyTAG` and
`list_supported_games()`. Unset it (or leave it unset) to fall back to the downloaded jar.

## Supported games

The following games are currently supported (as registered Gymnasium environments):

| Game | Gym ID | Obs type | Players |
|------|--------|----------|---------|
| Diamant | `TAG/Diamant-v0` | vector | 2+ |
| TicTacToe | `TAG/TicTacToe-v0` | vector | 2 |
| LoveLetter | `TAG/LoveLetter-v0` | vector | 2+ |
| Stratego | `TAG/Stratego-v0` | vector | 2 |
| SushiGo | `TAG/SushiGo-v0` | JSON | 2–5 |
| SushiGo (multi-agent) | `TAG/SushiGo-MA-v0` | JSON | 2 |
| PowerGrid | `TAG/PowerGrid-v0` | vector | 3–6 |
| Pandemic | `TAG/Pandemic-v0` | JSON text | 2–4 |

### Pandemic on Windows

Install Python 3, Git, Maven, and Java 21. From the root of a PyTAG clone, run this one PowerShell setup command:

```powershell
. .\scripts\setup_pandemic.ps1
```

The script creates `.venv`, installs PyTAG, clones TAG at commit `a5b8608cf53239b254f4e4a4f1b943c39794ddef`, applies `patches/pandemic-tag.patch`, and builds `target/TAG-pytag.jar`. It sets `PYTAG_JAR_PATH` and leaves this PowerShell session in `pytag/TabletopGames`, the required working directory for `data/pandemic`. Dot sourcing keeps those settings in the current shell. Run the same setup command in a new shell to restore them; it accepts an already patched checkout at the pinned commit. The jar and TAG checkout are generated locally and are ignored by Git.

Use the Python interpreter in `.venv` from that shell:

```python
from pytag import PyTAG

env = PyTAG(game_id="Pandemic")
observation, info = env.reset(seed=23)
action = info["legal_actions"][0]
observation, reward, done, info = env.step(action["id"])
```

The default has one Python player and one TAG random teammate, with JSON observations. `MultiAgentPyTAG(['python', 'python'], game_id='Pandemic', obs_type='json')` controls both. Each `reset()` and `step()` info object includes `legal_actions`, an ordered list of the current TAG legal actions. Each item has a deterministic string `id`, a readable `label`, structured `details`, and a `mask_index`. `step(item)`, `step(item['id'])`, and the existing `step(item['mask_index'])` all select that TAG action. The fixed mask has 131,072 slots; `mask_index` indexes that mask and may vary with an instance's action history, while `id` is derived only from the action's semantic key. Invalid Pandemic IDs raise `ValueError`.

TAG generates and executes every legal action, including movement, roles, event cards, discards, Forecast permutations, and reactions.
`env.get_action_trace()` returns the semantic IDs of actions selected by both Python and TAG-controlled players since the last reset. It is intended for evaluation logs; the agent observation remains the JSON returned by `reset()` and `step()`.

To run the 100-game random legal-action pilot twice in fresh Python processes (50 one-Python and 50 two-Python games per run), use this command after setup:

```powershell
& ..\..\.venv\Scripts\python.exe ..\..\scripts\validate_pandemic.py
```

The script changes each worker's working directory to the TAG checkout for `data/pandemic`. It writes a combined CSV, one CSV per run, and a differences CSV beside the selected output path. It compares outcomes, decision counts, full action traces, and exceptions for matching seeds; elapsed times are recorded but excluded from reproducibility comparisons. Each run uses seeds 0–49 for each player mode, and the command exits nonzero if any game fails.

The observation is a JSON string. It includes the public board, public hands and roles, counters, visible discards, deck counts, and the acting player's current phase. Infection cards visible during Forecast appear only in that phase. It does not include either unrevealed deck order. The city cube vector follows TAG's `yellow, red, blue, black` order. `gym.make('TAG/Pandemic-v0')` exposes the same JSON in a Gymnasium `Text` space for one Python player. The base environment contains no communication or disruption mechanics.

Current limits: the action mask is sparse and large; role/event combinations outside the seeded integration tests have not all been exercised. `reset(seed=...)` reseeds TAG's game setup, TAG's built-in random Pandemic teammate, and the Python action sampler. Team experiments should pin the TAG commit and patch and run separate processes for independent replicates. The TAG patch also corrects Contingency Planner Airlift, Dispatcher flights using the wrong hand, and missing terminal player results. The random legal-action pilot exercises losses; the controlled final-cure integration test exercises the win path. `playpandemic.py` is a legacy direct-JPype example; new agents should use `PyTAG`.

## Getting started

The `examples/` folder provides scripts to get started with the framework.
`pt-action-masking.py` demonstrates manual action masking; `gym-action-masking.py` extends this to a Gymnasium environment; `gym-random.py` uses the built-in action sampler; `ma-random.py` shows how to control multiple Python agents simultaneously.

The PPO baseline scripts (`ppo.py`, `ppo-lstm.py`, `ppo-selfplay.py`) reproduce the experiments from the papers listed below. `ppo-eval.py` loads a trained model for evaluation. The self-play script (`ppo-selfplay.py`) trains agents via self-play using `TAGSelfPlayGYm` and a checkpoint pool for opponent selection.

## Citing Information

If you use PyTAG in your work, please cite the relevant papers below.

The IEEE Transactions on Games journal paper covers the full framework including multiagent and self-play environments:
```bibtex
@article{balla2024pytag,
  author    = {Balla, Martin and Long, George E. M. and Goodman, James and Gaina, Raluca D. and Perez-Liebana, Diego},
  journal   = {IEEE Transactions on Games},
  title     = {{PyTAG}: Tabletop Games for Multiagent Reinforcement Learning},
  year      = {2024},
  volume    = {16},
  number    = {4},
  pages     = {993--1002},
  doi       = {10.1109/TG.2024.3404133}
}
```

The original CoG 2023 paper introduced the single-agent interface:
```bibtex
@inproceedings{balla2023pytag,
  author    = {Balla, Martin and Long, George E. M. and Jeurissen, Dominik and Goodman, James and Gaina, Raluca D. and Perez-Liebana, Diego},
  title     = {{PyTAG}: Challenges and Opportunities for Reinforcement Learning in Tabletop Games},
  booktitle = {IEEE Conference on Games (CoG)},
  year      = {2023}
}
```

To cite the TAG framework itself:
```bibtex
@inproceedings{gaina2020tag,
  author    = {Raluca D. Gaina and Martin Balla and Alexander Dockhorn and Raul Montoliu and Diego Perez-Liebana},
  title     = {{TAG}: A Tabletop Games Framework},
  booktitle = {Experimental AI in Games (EXAG), AIIDE 2020 Workshop},
  year      = {2020}
}
```

## Contact and contribute

The main method to contribute to our repository directly with code, or to suggest new features, point out bugs or ask
questions about the project is
through [creating new Issues on this github repository](https://github.com/GAIGResearch/TabletopGames/issues)
or [creating new Pull Requests](https://github.com/GAIGResearch/TabletopGames/pulls). Alternatively, you may contact the
authors of the papers listed above.

You can also find out more about the [QMUL Game AI Group](http://gameai.eecs.qmul.ac.uk/).

## Acknowledgements

This work was partly funded by the EPSRC CDT in Intelligent Games and Game Intelligence (IGGI)  EP/L015846/1 and EPSRC
research grant EP/T008962/1.
