#!/usr/bin/env bash
# Register W&B sweeps and print the agent command for each.
#
#   src/configs/sweeps/launch.sh all              every sweep in this folder
#   src/configs/sweeps/launch.sh component_compas monitor_ablation
#   WANDB_ENTITY=my-team src/configs/sweeps/launch.sh all
#
# Run from the repo root with the project venv active: the agent launches
# ${interpreter}, i.e. whatever python runs `wandb agent`. The configs are
# validated first, so a typo in a flag or scenario fails here rather than
# inside a cell hours into a grid.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
SWEEP_DIR=src/configs/sweeps
PROJECT=fado-ablations
ENTITY_ARGS=()
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  ENTITY_ARGS=(--entity "$WANDB_ENTITY")
fi

if [[ $# -eq 0 ]]; then
  echo "usage: $0 all | <name> [<name> ...]   (name = file without sweep_ and .yaml)" >&2
  ls "$SWEEP_DIR"/sweep_*.yaml | sed 's|.*/sweep_||; s|\.yaml$||' >&2
  exit 1
fi

if [[ "$1" == "all" ]]; then
  files=("$SWEEP_DIR"/sweep_*.yaml)
else
  files=()
  for name in "$@"; do
    files+=("$SWEEP_DIR/sweep_${name}.yaml")
  done
fi

python -m TESTS.check_sweeps

export PYTHONHASHSEED=0
for file in "${files[@]}"; do
  echo "== $file"
  out=$(wandb sweep --project "$PROJECT" ${ENTITY_ARGS[@]+"${ENTITY_ARGS[@]}"} "$file" 2>&1)
  echo "$out" | grep -E "Creating sweep|Run sweep agent with" || echo "$out"
done
