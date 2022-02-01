"""Runs the learner/evaluator."""

import os
import pickle
from pathlib import Path
import argparse
from matplotlib import animation, pyplot as plt

from tqdm import tqdm
import numpy as np
import tensorflow as tf

import common
import core_model
import cloth_model
from dataset import load_dataset_eval
from plot_cloth import plot_cloth


gpus = tf.config.experimental.list_physical_devices('GPU')
try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except RuntimeError as e:
    print(e)


model_dir = os.path.dirname(__file__)

def plot_timestep(timestep_data):
    figure = plt.figure()
    axis = figure.add_subplot(121, projection="3d")
    axis2 = figure.add_subplot(122, projection="3d")
    figure.set_figheight(6)
    figure.set_figwidth(12)

    skip = 10
    num_steps = timestep_data["pred_world_pos"].shape[0]
    num_frames = num_steps // skip
    upper_bound = np.amax(timestep_data["pred_world_pos"], axis=(0,1))
    lower_bound= np.amin(timestep_data["pred_world_pos"], axis=(0,1))
    def animate(frame):
        step = (frame*skip) % num_steps
        axis.cla()
        axis.set_xlim([lower_bound[0], upper_bound[0]])
        axis.set_ylim([lower_bound[1], upper_bound[1]])
        axis.set_zlim([lower_bound[2], upper_bound[2]])
        axis.autoscale(False)
        positions = timestep_data['pred_world_pos'][step]
        faces = timestep_data["cells"][step]
        axis.plot_trisurf(positions[:,0], positions[:, 1], faces, positions[:, 2])
        axis.set_title('Predicted')

        axis2.cla()
        axis2.set_xlim([lower_bound[0], upper_bound[0]])
        axis2.set_ylim([lower_bound[1], upper_bound[1]])
        axis2.set_zlim([lower_bound[2], upper_bound[2]])
        axis2.autoscale(False)
        positions = timestep_data['true_world_pos'][step]
        faces = timestep_data["cells"][step]
        axis2.plot_trisurf(positions[:,0], positions[:, 1], faces, positions[:, 2])
        axis2.set_title('Ground Truth')

        figure.suptitle(f"Time step: {step}")

        return figure,

    _ = animation.FuncAnimation(figure, animate, frames=num_frames, interval=100)

    # ani.save(filename)
    plt.show(block=True)

def frame_to_graph(frame, wind=False):
    """Builds input graph."""

    # construct graph nodes
    velocity = frame['world_pos'] - frame['prev|world_pos']
    node_type = tf.one_hot(frame['node_type'][:, 0], common.NodeType.SIZE)

    node_features = tf.concat([velocity, node_type], axis=-1)
    if wind:
        wind_velocities = tf.ones([len(velocity), len(frame['wind_velocity'])]) * frame['wind_velocity']
        node_features = tf.concat([node_features, wind_velocities], axis=-1)

    # construct graph edges
    senders, receivers = common.triangles_to_edges(frame['cells'])
    relative_world_pos = (tf.gather(frame['world_pos'], senders) -
                          tf.gather(frame['world_pos'], receivers))
    relative_mesh_pos = (tf.gather(frame['mesh_pos'], senders) -
                         tf.gather(frame['mesh_pos'], receivers))
    edge_features = tf.concat([
        relative_world_pos,
        tf.norm(relative_world_pos, axis=-1, keepdims=True),
        relative_mesh_pos,
        tf.norm(relative_mesh_pos, axis=-1, keepdims=True)], axis=-1)

    del frame['cells']

    return node_features, edge_features, senders, receivers, frame


def build_model(model, dataset, wind=False):
    """Initialize the model"""
    traj = next(iter(dataset))
    frame = {k: v[0] for k, v in traj.items()}
    node_features, edge_features, senders, receivers, frame = frame_to_graph(frame, wind=wind)
    graph = core_model.MultiGraph(node_features, edge_sets=[core_model.EdgeSet(edge_features, senders, receivers)])

    # call the model once to process all input shapes
    model.loss(graph, frame)

    # get the number of trainable parameters
    total = 0
    for var in model.trainable_weights:
        total += np.prod(var.shape)
    print(f'Total trainable parameters: {total}')


def rollout(model, initial_frame, num_steps, wind=False):
    """Rollout a model trajectory."""
    mask = tf.equal(initial_frame['node_type'], common.NodeType.NORMAL)

    prev_pos = initial_frame['prev|world_pos']
    curr_pos = initial_frame['world_pos']
    trajectory = []

    """In-situ visualization"""
    figure = plt.figure()
    axis = figure.add_subplot(121, projection="3d")
    figure.set_figheight(6)
    figure.set_figwidth(12)

    rollout_loop = tqdm(range(num_steps))
    for i in rollout_loop:
        frame = {**initial_frame, 'prev|world_pos': prev_pos, 'world_pos': curr_pos}
        node_features, edge_features, senders, receivers, frame = frame_to_graph(frame, wind=wind)
        graph = core_model.MultiGraph(node_features, edge_sets=[core_model.EdgeSet(edge_features, senders, receivers)])

        next_pos = model.predict(graph, frame)
        next_pos = tf.where(mask, next_pos, curr_pos)

        trajectory.append(curr_pos)

        """Viz"""
        upper_bound = np.amax(next_pos, axis=(0,1))
        lower_bound= np.amin(next_pos, axis=(0,1))

        axis.cla()
        axis.set_xlim([lower_bound[0], upper_bound[0]])
        axis.set_ylim([lower_bound[1], upper_bound[1]])
        axis.set_zlim([lower_bound[2], upper_bound[2]])
        axis.autoscale(False)
        positions = next_pos
        faces = frame['cells']
        axis.plot_trisurf(positions[:,0], positions[:, 1], faces, positions[:, 2])
        axis.set_title('Predicted')

        figure.suptitle(f"Time step: {i}")

        plt.show(block=True)

        prev_pos, curr_pos = curr_pos, next_pos

    return tf.stack(trajectory)


def to_numpy(t):
    """
    If t is a Tensor, convert it to a NumPy array; otherwise do nothing
    """
    try:
        return t.numpy()
    except:
        return t


def avg_rmse():
    results_path = os.path.join(model_dir, 'results')
    results_prefixes = ['og_long_noise-step9950000-loss0.05927.hdf5']

    for prefix in results_prefixes:
        all_errors = []
        for i in range(100):
            try:
                with open(os.path.join(results_path, prefix, f'{i:03d}.eval'), 'rb') as f:
                    data = pickle.load(f)
                    all_errors.append(data['errors'])
            except FileNotFoundError:
                continue

        keys = list(all_errors[0].keys())
        all_errors = {k: np.array([errors[k] for errors in all_errors]) for k in keys}

        for k, v in all_errors.items():
            print(prefix, k, np.mean(v))


def evaluate(checkpoint_file, dataset_path, num_trajectories, wind=False):
    dataset = load_dataset_eval(
        path=dataset_path,
        split='test',
        fields=['world_pos'],
        add_history=True
    )

    model = core_model.EncodeProcessDecode(
        output_dims=3,
        embed_dims=128,
        num_layers=3,
        num_iterations=15,
        num_edge_types=1
    )
    model = cloth_model.ClothModel(model)

    # build the model
    build_model(model, dataset, wind=wind)

    model.load_weights(checkpoint_file, by_name=True)

    Path(os.path.join(model_dir, 'results')).mkdir(exist_ok=True)
    for i, trajectory in enumerate(dataset.take(num_trajectories)):
        initial_frame = {k: v[0] for k, v in trajectory.items()}
        predicted_trajectory = rollout(model, initial_frame, trajectory['cells'].shape[0], wind=wind)

        error = tf.reduce_mean(tf.square(predicted_trajectory - trajectory['world_pos']), axis=-1)
        rmse_errors = {f'{horizon}_step_error': tf.math.sqrt(tf.reduce_mean(error[1:horizon + 1])).numpy()
                       for horizon in [1, 10, 20, 50, 100, 200, 398]}
        print(f'RMSE Errors: {rmse_errors}')

        data = {**trajectory, 'true_world_pos': trajectory['world_pos'], 'pred_world_pos': predicted_trajectory, 'errors': rmse_errors}
        data = {k: to_numpy(v) for k, v in data.items()}
        plot_cloth(data)

        save_path = os.path.join(model_dir, 'results', f'{i:03d}.eval')
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
            print(f'Evaluation results saved in {save_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="Path to checkpoint file")
    parser.add_argument("--data_path", help="Path to dataset")
    parser.add_argument("--num_trajectories", type=int, help="Number of trajectories to evaluate")
    parser.add_argument("--wind", "-w", action="store_true", help="Toggle for model to evaluate using diffrent wind velocities")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.data_path, args.num_trajectories, wind=args.wind)


if __name__ == '__main__':
    main()