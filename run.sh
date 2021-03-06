#!/bin/bash
# Copyright 2020 Deepmind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Fail on any error.
set -e

# Display commands being run.
set -x

# Install dependencies.
pip install --upgrade -r requirements.txt

DATA_DIR="./datasets/cylinder_flow"

# Train for a few steps.
CHECKPOINT="${DATA_DIR}/checkpoints/flag-simple_weights-step2100000-loss0.0680.hdf5"
# python -m train_cloth --data_path=${DATA_DIR} --num_steps=100000
python -m train_cfd --data_path=${DATA_DIR} --num_steps=2000000

# Generate a rollout trajectory
# ROLLOUT_PATH="${DATA_DIR}/rollout.pkl"
# python -m cloth_eval --checkpoint=${CHECKPOINT} --data_path=${DATA_DIR} --num_trajectories=2

EVAL_RESULT="${DATA_DIR}/results/flag-simple_weights-step2100000-loss0.0680.hdf5/000.eval"
# Plot the rollout trajectory
# python -m plot_cloth --datafile=${EVAL_RESULT}

echo "Test run complete."
