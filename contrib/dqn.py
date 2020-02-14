# RL Agent that plays 2048 using DQN.
from collections import deque
from envs.nick_2048 import Nick2048
import gym
import logging
import mlflow.tracking
import numpy as np
import os
import pandas as pd
import random
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
import tensorflow as tf
import tensorflow_probability as tfp
from ray.tune import Trainable

from strategies.utility import softmax

#logging.basicConfig(level=logging.DEBUG)


class Memory:
    def __init__(self, maxlen=10000):
        self.states = deque(maxlen=maxlen)
        self.actions = deque(maxlen=maxlen)
        self.rewards = deque(maxlen=maxlen)
        self.dones = deque(maxlen=maxlen)
        self.next_states = deque(maxlen=maxlen)

    def append(self, t):
        assert len(t) == 5
        self.states.append(t[0])
        self.actions.append(t[1])
        self.rewards.append(t[2])
        self.dones.append(t[3])
        self.next_states.append(t[4])

    def get_random_batch(self, batchsize):
        assert batchsize <= len(self.states)
        candidate_indices = list(range(len(self.states)))
        random.shuffle(candidate_indices)
        states, actions, rewards, dones, next_states = [], [], [], [], []
        for i in range(batchsize):
            states.append(self.states[candidate_indices[i]])
            actions.append(self.actions[candidate_indices[i]])
            rewards.append(self.rewards[candidate_indices[i]])
            dones.append(self.dones[candidate_indices[i]])
            next_states.append(self.next_states[candidate_indices[i]])
        return (np.asarray(states),
                np.asarray(actions),
                np.asarray(rewards).reshape((batchsize, 1)),
                np.asarray(dones).reshape((batchsize, 1)),
                np.asarray(next_states))


class DQN(Trainable):
    def _setup(self, config):
        self.params = config
        self.mlflow_client = mlflow.tracking.MlflowClient()
        self.mlflow_run = self.mlflow_client.create_run(experiment_id="0")
        self.mlflow_log_params(config)
        self.env = Nick2048()
        self.q_models = []
        q_model = keras.Sequential(
            [
                keras.layers.Dense(20, activation="relu"),
                keras.layers.Dense(20, activation="relu"),
                keras.layers.Dense(20, activation="relu"),
                keras.layers.Dense(1),
            ]
        )
        for _ in range(self.env.action_space.n):
            self.q_models.append(keras.models.clone_model(q_model))
        [m.build(input_shape=[1, self.env.observation_space.shape[0]]) for m in self.q_models]
        self.loss_fn = keras.losses.mean_squared_error
        self.optimizer = keras.optimizers.Adam(lr=self.params["learning_rate"])
        self.memory = Memory(self.params["buffer_size"])

    def mlflow_log_metric(self, key, val, timestamp=None, step=None):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, val, timestamp, step)

    def mlflow_log_params(self, params):
        for k, v in params.items():
            self.mlflow_client.log_param(self.mlflow_run.info.run_id, k, v)

    def _save(self, tmp_checkpoint_dir):
        raise NotImplementedError
        #TODO port to save all action models
        #self.q_model.save(os.path.join(tmp_checkpoint_dir, "/model"))

    def _restore(self, tmp_checkpoint_dir):
        raise NotImplementedError
        #TODO port to restore all action models
        #load_model(os.path.join(tmp_checkpoint_dir, "/model"))

    def get_action(self, episode_num, state):
        prob_random_action = min(1., (self.params["num_init_random_actions"] + self.params["epsilon"]) / (episode_num + 1.))
        self.mlflow_log_metric("prob random action", prob_random_action, step=episode_num)
        if random.random() < prob_random_action:
            action = self.env.action_space.sample()
        else:
            q_vals = tf.concat([self.q_models[a](np.asarray(self.env.get_afterstate(state, a)[0])[np.newaxis])
                                for a in range(self.env.action_space.n)], 0)
            action = tf.squeeze(tf.argmax(q_vals)).numpy()
        return action

    def _train(self):
        train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
        game_scores = []
        game_num_steps = []
        for episode_num in range(self.params["num_episodes"]):
            alpha = self.params["alpha0_ie_init_step_size"] / (1 + episode_num * self.params["alpha_decay"])
            logging.debug(f"alpha: {alpha}")
            self.mlflow_log_metric("alpha", alpha)
            state = self.env.reset()
            game_score = 0
            for step_num in range(self.params["max_steps_per_episode"]):
                action = self.get_action(episode_num, state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.asarray(next_state)
                self.memory.append((state, action, reward, done, next_state))
                state = next_state
                game_score += reward
                if done:
                    break

            if episode_num >= self.params["learning_starts"]:
                states, actions, rewards, dones, next_states = self.memory.get_random_batch(self.params["batch_size"])
                    # batch = get batch of states, actions, rewards, dones, next_states
                    # for a in env.action_space.n:
                    #     b = [t for t in batch if t[1] == a]
                    #     q = q_models[actions](states)
                    #     target_q = (1-alpha) q(s,action) + alpha (r + gamma * argmax_a(q_models[a](s')))
                    #     fit model to minimize loss(y, target_y)
                for a in range(self.env.action_space.n):
                    if any(actions.squeeze() == a):
                        s = states[actions.squeeze() == a]
                        afterstates = np.asarray([self.env.get_afterstate(st, a)[0] for st in s]).reshape(s.shape)
                        #canonicals = [self.env.get_canonical_board(st) for st in afterstates]
                        r = rewards[actions.squeeze() == a]
                        d = dones[actions.squeeze() == a]
                        s_p = next_states[actions.squeeze() == a]
                        with tf.GradientTape() as tape:
                            q_vals = self.q_models[a](afterstates)
                            next_ca = []
                            # this mess is necessary since we are using afterstates
                            for n_a in range(self.env.action_space.n):
                                next_as = np.asarray([self.env.get_afterstate(st, n_a)[0] for st in s_p]).reshape(s_p.shape)
                                #next_canonicals = [self.env.get_canonical_board(st) for st in next_as]
                                next_ca.append(self.q_models[n_a](next_as))
                            next_q_vals_all = tf.concat(next_ca, 1)
                            next_q_vals_simple = tf.reduce_max(next_q_vals_all, axis=1)
                            next_q_vals = tf.expand_dims(next_q_vals_simple, 1)
                            disc_next_q = (1 - d) * self.params["gamma_ie_discount_rate"] * next_q_vals
                            td_target = r + disc_next_q
                            q_val_targets = (1 - alpha) * q_vals + alpha * td_target
                            loss = self.loss_fn(q_vals, q_val_targets)
                            grads = tape.gradient(loss, self.q_models[a].trainable_variables)
                            self.optimizer.apply_gradients(zip(grads, self.q_models[a].trainable_variables))
                            logging.debug(f"r: {r}")
                            logging.debug(f"d: {d}")
                            logging.debug(f"next_q_vals_all: {next_q_vals_all}")
                            logging.debug(f"next_q_vals: {next_q_vals}")
                            logging.debug(f"td_target: {td_target}")
                            logging.debug(f"q_val_targets: {q_val_targets}")
                        train_acc_metric.update_state(q_val_targets, q_vals)

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
                % (
                    step_num + 1,
                    episode_num,
                    game_score,
                    avg_game_score,
                    avg_last_30,
                )
            )
            self.mlflow_log_metric("game score", game_score, step=episode_num + step_num)
            self.mlflow_log_metric("avg game score", avg_game_score, step=episode_num + step_num)
            self.mlflow_log_metric("avg_score_last_30", avg_last_30, step=episode_num + step_num)
            self.mlflow_log_metric("game num steps", step_num + 1, step=episode_num + step_num)
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
    params["num_episodes"] = 1000
    params["epsilon"] = 0.5
    params["num_init_random_actions"] = 1 #20
    params["max_steps_per_episode"] = 500
    params["alpha0_ie_init_step_size"] = 0.95
    params["alpha_decay"] = 0.00005
    params["gamma_ie_discount_rate"] = 0.9
    params["learning_rate"] = 0.01
    params["learning_starts"] = 3 #30
    params["batch_size"] = 3 #30
    params["buffer_size"] = 10000
    # As a heuristic, make sure we have enough data before we start learning
    assert params["learning_starts"] >= params["batch_size"]
    dqn = DQN(params)
    dqn.train()
