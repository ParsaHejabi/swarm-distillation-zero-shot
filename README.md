# Prompt Consistency for Zero-Shot Task Generalization
This is the official implementation of the [paper](https://arxiv.org/abs/2205.00049):

```
Prompt Consistency for Zero-Shot Task Generalization
Chunting Zhou*, Junxian He*, Xuezhe Ma, Taylor Berg-Kirkpatrick, Graham Neubig
Preprint 2022
```

## Dependencies

This repo is a fork of the [huggingface transformers](https://github.com/huggingface/transformers) repo (forked on Oct 29, 2021), and the code is tested on [PyTorch](https://pytorch.org) 1.9.0. Please follow the instructions below to install dependencies after you set up PyTorch:

```bash
git clone git@github.com:violet-zct/swarm-distillation-zero-shot.git
cd swarm-distillation-zero-shot

# install transformers from this repo
pip install -e .

# install other requirements
pip install datasets
```

## Usage
We are still working on cleaning the code, for early usage please refer to `exps/ttt/final_3B.sh` for an example training script that we used to tune the T0-3B model.
For running on a single machine with multiple GPUs and without Slurm, see
`exps_ttt/run_deepspeed_local.sh`. Run this script from anywhere and it will
launch training with DeepSpeed on 8 GPUs. The script also adds the repository
root to `PYTHONPATH` so that local modules like `ttt` can be imported. By
default, the script allows Transformers to download models if they are not
cached locally. Uncomment the `TRANSFORMERS_OFFLINE` line in the script if you
need to run strictly offline.

If you want to debug the training loop without DeepSpeed, use
`exps_ttt/run_local.sh`. This script mirrors the settings of the DeepSpeed
version but runs the training script directly with PyTorch so that variables can
be inspected easily. It defaults to the smaller `T0_3B` model so that debugging
can be done on a single GPU. The script also sets `HF_ENDPOINT` to
`https://huggingface.co` to avoid errors caused by inheriting a wrong endpoint
value from the environment.

If you see errors about protobuf descriptors when launching the script, set the
environment variable `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` or
install a `protobuf` version <=3.20 to resolve the issue.

## Citation

```
@article{zhou2022prompt,
  title={Prompt Consistency for Zero-Shot Task Generalization},
  author={Chunting Zhou and Junxian He and Xuezhe Ma and Taylor Berg-Kirkpatrick and Graham Neubig},
  journal={ArXiv},
  year={2022},
  volume={abs/2205.00049}
}
```