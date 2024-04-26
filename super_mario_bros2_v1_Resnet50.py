"""
Super-Mario-Bros2-v1 -- ResNet50
"""

import time
import os
import random
from collections import deque

import gym
import gym_super_mario_bros.actions as actions
import numpy as np
import keras
from keras.models import Sequential, clone_model
from keras.layers import Conv2D, Dense, Flatten
from keras.optimizers import Adam
from wrappers import wrap_nes

from keras.layers import Input, Flatten, Dense
from keras.optimizers import Adam
from keras.applications import ResNet50
from keras.models import Model


class ReplyBuffer:
    def __init__(self, memory_size=20000):
        self.state = deque(maxlen=memory_size)
        self.action = deque(maxlen=memory_size)
        self.reward = deque(maxlen=memory_size)
        self.next_state = deque(maxlen=memory_size)
        self.done= deque(maxlen=memory_size)

    def append(self, state, action, reward, next_state, done):
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)
        self.next_state.append(next_state)
        self.done.append(done)

    def __len__(self):
        return len(self.done)


class Agent:
    def __init__(self, env, memory_size=20000):
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.observation_shape = env.observation_space.shape
        self.memory = ReplyBuffer(memory_size=memory_size)
        self.batch_size = 32
        self.update_frequency = 4
        self.tau = 1000
        self.gamma = 0.99  # discount rate
        self.epsilon = 1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999
        self.learning_rate = 0.0001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())

    def _build_model(self):
        """Builds a ResNet50 model for the Super Mario Bros agent."""
        input_shape = self.observation_shape  # Assuming self.observation_shape is already set
        num_actions = self.action_size  # Assuming self.action_size is already set
        learning_rate = self.learning_rate  # Assuming self.learning_rate is already set
        print("Input shape:", input_shape)
        base_model = ResNet50(include_top=False, input_shape=input_shape, weights=None)
        x = base_model.output
        x = Flatten()(x)
        x = Dense(512, activation='elu', kernel_initializer='random_uniform')(x)
        output = Dense(num_actions, activation='softmax', name='output')(x)

        model = Model(inputs=base_model.input, outputs=output)
        model.compile(optimizer=Adam(lr=learning_rate), loss='mse')
        return model

    def update_target_network(self):
        self.target_model = clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append(state, action, reward, next_state, done)

    def act(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state)[0])

    def experience_reply(self):
        if self.batch_size > len(self.memory):
            return

        # Get indices of samples for replay buffers
        indices = np.random.choice(range(len(self.memory)), size=self.batch_size)

        # Randomly sample a batch from the memory
        state_sample = np.array([self.memory.state[i][0] for i in indices])
        action_sample = np.array([self.memory.action[i] for i in indices])
        reward_sample = np.array([self.memory.reward[i] for i in indices])
        next_state_sample = np.array([self.memory.next_state[i][0] for i in indices])
        done_sample = np.array([self.memory.done[i] for i in indices])

        # Batch prediction to save speed
        target = self.model.predict(state_sample)
        target_next = self.target_model(next_state_sample)

        for i in range(self.batch_size):
            if done_sample[i]:
                target[i][action_sample[i]] = reward_sample[i]
            else:
                target[i][action_sample[i]] = reward_sample[i] + self.gamma * (np.amax(target_next[i]))

        self.model.fit(
            np.array(state_sample),
            np.array(target),
            batch_size=self.batch_size,
            verbose=0
        )

    def load_weights(self, weights_file):
        self.epsilon = self.epsilon_min
        self.model.load_weights(weights_file)

    def save_weights(self, weights_file):
        self.model.save_weights(weights_file)


if __name__ == "__main__":
    """
    Main program
    """
    monitor = False

    # Initializes the environment
    env = wrap_nes("SuperMarioBros-v1", actions.SIMPLE_MOVEMENT)

    # Records the environment
    if monitor:
        env = gym.wrappers.Monitor(env, "recording", video_callable=lambda episode_id: True, force=True)

    # Defines training related constants
    num_episodes = 50000
    num_episode_steps = env.spec.max_episode_steps  # constant value
    frame_count = 0
    max_reward = 0

    # Create lists to store metrics
    avg_rewards = []
    avg_game_scores = []
    avg_steps_per_episode = []
    training_times = []

    # Creates an agent
    agent = Agent(env=env, memory_size=20000)

    # Loads the weights
    if os.path.isfile("super_mario_bros2_v1_resnet50.h5"):
        agent.load_weights("super_mario_bros2_v1_resnet50.h5")

    for episode in range(num_episodes):
        # Defines the total reward per episode
        total_reward = 0

        # Resets the environment
        observation = env.reset()
        
        # Gets the state
        state = np.reshape(observation, (1,) + env.observation_space.shape)
        start_time = time.time()
        for episode_step in range(num_episode_steps):
            # Renders the screen after new environment observation
            env.render(mode="human")

            # Gets a new action
            action = agent.act(state)

            # Takes action and calculates the total reward
            observation, reward, done, _ = env.step(action)
            total_reward += reward
            
            # print("Input shape:", env.observation_space.shape)
            # Gets the next state
            next_state = np.reshape(observation, (1,) + env.observation_space.shape)

            # Memorizes the experience
            agent.memorize(state, action, reward, next_state, done)

            # Updates the online network weights
            if frame_count % agent.update_frequency == 0:
                agent.experience_reply()

            # Updates the target network weights
            if frame_count % agent.tau == 0:
                agent.update_target_network()

            # Updates the state
            state = next_state

            # Updates the total steps
            frame_count += 1

            if done:
                print("Episode %d/%d finished after %d episode steps with total reward = %f."
                      % (episode + 1, num_episodes, episode_step + 1, total_reward))
                break

            elif episode_step >= num_episode_steps - 1:
                print("Episode %d/%d timed out at %d with total reward = %f."
                      % (episode + 1, num_episodes, episode_step + 1, total_reward))

        # Track episode end time and calculate episode duration
        end_time = time.time()
        episode_train_time = end_time - start_time

        # Calculate average reward, game score, and steps per episode
        avg_reward = total_reward / num_episodes
        avg_game_score = total_reward  # Assuming game score is equal to total reward
        avg_steps = num_episode_steps/num_episodes

        # Append metrics to lists
        avg_rewards.append(avg_reward)
        avg_game_scores.append(avg_game_score)
        avg_steps_per_episode.append(avg_steps)
        training_times.append(episode_train_time)

        # Print episode metrics
        print(f"Episode {episode + 1}/{num_episodes}: Avg. Reward={avg_reward}, "
              f"Avg. Game Score={avg_game_score}, Avg. Steps={avg_steps}, "
                f"Training Time={episode_train_time} sec"
              )

        # Updates the epsilon value
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        # Saves the online network weights
        if total_reward > max_reward:
            agent.save_weights("super_mario_bros2_v1_resnet50.h5")
            keras.backend.clear_session()

    # Closes the environment
    env.close()

    # Calculate average training time per episode
    avg_training_time = np.mean(training_times)
    print(f"Average Training Time per Episode: {avg_training_time} sec")

    # Calculate average metrics over all episodes
    avg_reward_overall = np.mean(avg_rewards)
    avg_game_score_overall = np.mean(avg_game_scores)
    avg_steps_overall = np.mean(avg_steps_per_episode)

    # save avg metrics to a csv file
    with open("res_avg_rewards.csv", "w") as f:
        for reward in avg_rewards:
            f.write(f"{reward}\n")

    with open("res_avg_game_scores.csv", "w") as f:
        for game_score in avg_game_scores:
            f.write(f"{game_score}\n")

    with open("res_avg_steps_per_episode.csv", "w") as f:
        for steps in avg_steps_per_episode:
            f.write(f"{steps}\n")

    with open("res_training_times.csv", "w") as f:
        for time in training_times:
            f.write(f"{time}\n")
    
    # Print average metrics over all episodes
    with open("res_avg_metrics.csv", "w") as f:
        # column headers
        f.write("Average Reward, Average Game Score, Average Steps per Episode, Training Time\n")
        # data
        f.write(f"{avg_reward_overall}, {avg_game_score_overall}, {avg_steps_overall}, {avg_training_time}\n")    

    print(f"Average Reward over {num_episodes} episodes: {avg_reward_overall}")
    print(f"Average Game Score over {num_episodes} episodes: {avg_game_score_overall}")
    print(f"Average Steps per Episode over {num_episodes} episodes: {avg_steps_overall}")