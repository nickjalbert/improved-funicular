# RL Agent that plays 2048 using Deep Value Network (DVN) -- a modification of DQN.
from collections import deque
from envs.nick_2048 import Nick2048
import logging
import mlflow.tracking
import numpy as np
import os
import random
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
import tensorflow as tf
from ray.tune import Trainable


# logging.basicConfig(level=logging.DEBUG)

class Memory:
    def __init__(self, maxlen=10000):
        self.states = deque(maxlen=maxlen)
        self.actions = deque(maxlen=maxlen)
        self.afterstates = deque(maxlen=maxlen)
        self.rewards = deque(maxlen=maxlen)
        self.dones = deque(maxlen=maxlen)
        self.next_states = deque(maxlen=maxlen)

    def append(self, t):
        assert len(t) == 6
        self.states.append(t[0])
        self.actions.append(t[1])
        self.afterstates.append(t[2])
        self.rewards.append(t[3])
        self.dones.append(t[4])
        self.next_states.append(t[5])

    def get_random_batch(self, batchsize):
        assert batchsize <= len(self.states)
        candidate_indices = list(range(len(self.states)))
        random.shuffle(candidate_indices)
        states, actions, afterstates, rewards, dones, next_states = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for i in range(batchsize):
            states.append(self.states[candidate_indices[i]])
            actions.append(self.actions[candidate_indices[i]])
            afterstates.append(self.afterstates[candidate_indices[i]])
            rewards.append(self.rewards[candidate_indices[i]])
            dones.append(self.dones[candidate_indices[i]])
            next_states.append(self.next_states[candidate_indices[i]])
        return (
            np.asarray(states),
            np.asarray(actions),
            np.asarray(afterstates),
            np.asarray(rewards).reshape((batchsize, 1)),
            np.asarray(dones).reshape((batchsize, 1)),
            np.asarray(next_states),
        )


class DVN(Trainable):
    def _setup(self, config):
        self.params = config
        self.mlflow_client = mlflow.tracking.MlflowClient()
        self.mlflow_run = self.mlflow_client.create_run(experiment_id="0")
        self.mlflow_log_params(config)
        self.env = Nick2048()
        self.v_model = keras.Sequential(
            [
                keras.layers.Dense(20, activation="relu"),
                keras.layers.Dense(20, activation="relu"),
                keras.layers.Dense(20, activation="relu"),
                keras.layers.Dense(1),
            ]
        )
        self.v_model.build(input_shape=[1, self.env.observation_space.shape[0]])
        self.loss_fn = keras.losses.mean_squared_error
        self.optimizer = keras.optimizers.Adam(lr=self.params["learning_rate"])
        self.memory = Memory(self.params["buffer_size"])

    def mlflow_log_metric(self, key, val, timestamp=None, step=None):
        self.mlflow_client.log_metric(
            self.mlflow_run.info.run_id, key, val, timestamp, step
        )

    def mlflow_log_params(self, params):
        for k, v in params.items():
            self.mlflow_client.log_param(self.mlflow_run.info.run_id, k, v)

    def _save(self, tmp_checkpoint_dir):
        self.v_model.save(os.path.join(tmp_checkpoint_dir, "/model"))

    def _restore(self, tmp_checkpoint_dir):
        self.v_model = load_model(os.path.join(tmp_checkpoint_dir, "/model"))

    def get_action(self, episode_num, state):
        prob_random_action = min(
            1.0,
            (self.params["num_init_random_episodes"] + self.params["epsilon"])
            / (episode_num + 1.0),
        )
        self.mlflow_log_metric(
            "prob random action", prob_random_action, step=episode_num
        )
        if random.random() < prob_random_action:
            action = self.env.action_space.sample()
        else:
            state_vals = tf.concat(
                [
                    self.v_model(
                        np.asarray(
                            self.env.get_canonical_board(
                                self.env.get_afterstate(state, a)[0]
                            )
                        )[np.newaxis]
                    )
                    for a in range(self.env.action_space.n)
                ],
                0,
            )
            action = tf.squeeze(tf.argmax(state_vals)).numpy()
        return action

    def _train(self):
        train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
        game_scores = []
        game_num_steps = []
        for episode_num in range(self.params["num_episodes"]):
            alpha = self.params["alpha0_ie_init_step_size"] / (
                1 + episode_num * self.params["alpha_decay"]
            )
            logging.debug(f"alpha: {alpha}")
            self.mlflow_log_metric("alpha", alpha)
            state = self.env.reset()
            game_score = 0
            for step_num in range(self.params["max_steps_per_episode"]):
                action = self.get_action(episode_num, state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.asarray(next_state)
                self.memory.append(
                    (
                        state,
                        action,
                        self.env.get_canonical_board(
                            self.env.get_afterstate(state, action)[0]
                        ),
                        reward,
                        done,
                        next_state,
                    )
                )
                state = next_state
                game_score += reward
                if done:
                    break

            if episode_num >= self.params["learning_starts"]:
                (
                    states,
                    actions,
                    afterstates,
                    rewards,
                    dones,
                    next_states,
                ) = self.memory.get_random_batch(self.params["batch_size"])
                with tf.GradientTape() as tape:
                    vals = self.v_model(afterstates)
                    next_ca = []  # next_canonical_afterstates
                    for n_a in range(self.env.action_space.n):  # n_a is next_action
                        next_as = [
                            self.env.get_afterstate(s, n_a)[0] for s in next_states
                        ]
                        next_canonicals = np.asarray(
                            [self.env.get_canonical_board(s) for s in next_as]
                        ).reshape(next_states.shape)
                        next_ca.append(self.v_model(next_canonicals))
                    next_vals_all = tf.concat(next_ca, 1)
                    next_vals = tf.expand_dims(tf.reduce_max(next_vals_all, axis=1), 1)
                    disc_next_v = (
                        (1 - dones) * self.params["gamma_ie_discount_rate"] * next_vals
                    )
                    td_target = rewards + disc_next_v
                    val_targets = (1 - alpha) * vals + alpha * td_target
                    loss = self.loss_fn(vals, val_targets)
                    grads = tape.gradient(loss, self.v_model.trainable_variables)
                    self.optimizer.apply_gradients(
                        zip(grads, self.v_model.trainable_variables)
                    )
                    logging.debug(f"rewards: {rewards}")
                    logging.debug(f"dones: {dones}")
                    logging.debug(f"next_vals_all: {next_vals_all}")
                    logging.debug(f"next_vals: {next_vals}")
                    logging.debug(f"td_target: {td_target}")
                    logging.debug(f"val_targets: {val_targets}")
                train_acc_metric.update_state(val_targets, vals)

                logging.debug(
                    f"accuracy in episode {episode_num}: {train_acc_metric.result().numpy()}"
                )
                train_acc_metric.reset_states()

            game_scores.append(game_score)
            game_num_steps.append(step_num + 1)
            avg_game_score = np.mean(game_scores)
            avg_last_30 = np.mean(game_scores[-30:])
            print(
                "%s steps in episode %s, score: %s, running_avg: %.0f, avg_last_30_games: %.0f"
                % (step_num + 1, episode_num, game_score, avg_game_score, avg_last_30,)
            )
            self.mlflow_log_metric("game score", game_score, step=episode_num)
            self.mlflow_log_metric("avg game score", avg_game_score, step=episode_num)
            self.mlflow_log_metric("avg_score_last_30", avg_last_30, step=episode_num)
            self.mlflow_log_metric("game num steps", step_num + 1, step=episode_num)
            self.mlflow_log_metric(
                "avg num steps", np.mean(game_num_steps), step=episode_num
            )
        return {
            "avg_game_score": avg_game_score,
            "avg_num_steps": np.mean(game_num_steps),
            "episodes_total": episode_num + 1,
            "timesteps_total": np.sum(game_num_steps),
        }


if __name__ == "__main__":
    params = {}
    params["num_episodes"] = 1000000
    params["epsilon"] = 0.5
    params["num_init_random_episodes"] = 50
    params["max_steps_per_episode"] = 500
    params["alpha0_ie_init_step_size"] = 0.95
    params["alpha_decay"] = 0.00005
    params["gamma_ie_discount_rate"] = 0.9
    params["learning_rate"] = 0.01
    params["learning_starts"] = 50
    params["batch_size"] = 30
    params["buffer_size"] = 500000
    # As a heuristic, make sure we have enough data before we start learning
    assert params["learning_starts"] >= params["batch_size"]
    dvn = DVN(params)
    dvn.train()
