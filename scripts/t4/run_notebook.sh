#!/bin/bash
# Execute notebooks/colab_free_tier.ipynb top to bottom on the launch.sh instance, from a cold start like a fresh
# Colab runtime: empty Hugging Face cache, a new Python 3.12 venv with only jupyter + papermill, and the notebook's
# own %pip install of hev-rerank from GitHub at REF. Copies the executed notebook back.
# Usage: scripts/t4/run_notebook.sh REF   (STATE as for launch.sh)
set -eu
ref=${1:?git ref of hev-rerank to install}; state=${STATE:-.t4-state}; . "$state/env"
ssh_() { ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -i "$state/key.pem" ubuntu@$ip "$@"; }
scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -i "$state/key.pem" notebooks/colab_free_tier.ipynb ubuntu@$ip:nb_in.ipynb
ssh_ "set -e; rm -rf ~/.cache/huggingface ~/nbvenv
  command -v ~/.local/bin/uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
  ~/.local/bin/uv venv -q ~/nbvenv --python 3.12 && ~/.local/bin/uv pip install -q --python ~/nbvenv/bin/python pip papermill ipykernel
  ~/nbvenv/bin/python -m ipykernel install --user --name nbvenv >/dev/null
  # The DLAMI exports its system CUDA libraries in LD_LIBRARY_PATH; they clash with the pip-installed cuDNN (Colab has no such override).
  cd ~ && /usr/bin/time -f 'papermill wall %e s' env -u LD_LIBRARY_PATH ~/nbvenv/bin/papermill nb_in.ipynb nb_out.ipynb -k nbvenv -p HEV_RERANK_REF $ref --log-output 2>&1 | grep -v 'Loading weights\|Fetching' | tail -40"
scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -i "$state/key.pem" ubuntu@$ip:nb_out.ipynb notebooks/colab_free_tier.ipynb
