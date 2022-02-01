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

DATA_DIR="./datasets/flag_simple"

# Train for a few steps.
CHK_DIR="${DATA_DIR}/checkpoints"
python -m train_cloth --data_path=${DATA_DIR} --num_steps=100000

# Generate a rollout trajectory
ROLLOUT_PATH="${DATA_DIR}/rollout.pkl"
# python -m plot_cloth data_path=${DATA_DIR} --rollout_path=${ROLLOUT_PATH} --num_rollouts=4

# Plot the rollout trajectory
# python -m plot_cloth --rollout_path=${ROLLOUT_PATH}

echo "Test run complete."
