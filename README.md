# Learning Mesh-Based Simulation with Graph Networks (ICLR 2021)

This code is taken from DeepMind's MeshGraphNets [repository](https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets). 

Video site: [sites.google.com/view/meshgraphnets](https://sites.google.com/view/meshgraphnets)

Paper: [arxiv.org/abs/2010.03409](https://arxiv.org/abs/2010.03409)

Citation:

    @inproceedings{pfaff2021learning,
      title={Learning Mesh-Based Simulation with Graph Networks},
      author={Tobias Pfaff and
              Meire Fortunato and
              Alvaro Sanchez-Gonzalez and
              Peter W. Battaglia},
      booktitle={International Conference on Learning Representations},
      year={2021}
    }

## Overview

This release contains the full datasets used in the paper, as well as data
loaders (dataset.py), and the learned model core (core_model.py).
These components are designed to work with all of our domains.

This folder also include demonstration code for a full training and evaluation pipeline,
for the `cylinder_flow` and `flag_simple` domains only. This
includes graph encoding, evaluation, rollout and plotting trajectory.
Refer to the respective `cfd_*` and `cloth_*` files for details.

## Setup

Download a dataset by going to the root folder of this repository and do:

    sh get-datasets.sh

Prepare environment, install dependencies:

    conda create --name ${ENV} python=3.6
    conda activate ${ENV}
    pip install -r requirements.txt

## Running the model

Train a model:

    python -m run_model --mode=train --model=cloth \
        --checkpoint_dir=datasets/checkpoints --dataset_dir=datasets/flag_simple

Generate some trajectory rollouts:

    python -m run_model --mode=eval --model=cloth \
        --checkpoint_dir=datasets/checkpoints --dataset_dir=datasets/flag_simple \
        --rollout_path=datasets/rollouts/rollout_flag.pkl

Plot a trajectory:

    python -m plot_cloth --rollout_path=datasets/rollouts/rollout_flag.pkl

The instructions above train a model for the `flag_simple` domain; for
the `cylinder_flow` dataset, use `--model=cfd` and the `plot_cfd` script.

## Datasets

Datasets can be downloaded using the script `get-datasets.sh`. They contain
a metadata file describing the available fields and their shape, and tfrecord
datasets for train, valid and test splits.
Dataset names match the naming in the paper.
The following datasets are available:

    airfoil
    cylinder_flow
    deforming_plate
    flag_minimal
    flag_simple
    flag_dynamic
    flag_dynamic_sizing
    sphere_simple
    sphere_dynamic
    sphere_dynamic_sizing

`flag_minimal` is a truncated version of flag_simple, and is only used for
integration tests. `flag_dynamic_sizing` and `sphere_dynamic_sizing` can be
used to learn the sizing field. These datasets have the same structure as
the other datasets, but contain the meshes in their state before remeshing,
and define a matching `sizing_field` target for each mesh.
