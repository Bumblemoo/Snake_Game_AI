import torch
import random
import numpy as np
from collections import deque
from game import SnakeGame, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot


BLOCK_SIZE = 20

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # Control randomness
        self.gamma = 0  # Discount Rate
        self.memory = deque(
            maxlen=MAX_MEMORY
        )  # If we exceed MAX_MEMORY, then it will automatically call popleft() and remove from the queue.

        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y + BLOCK_SIZE)

        dir_r = game.direction == Direction.RIGHT
        dir_d = game.direction == Direction.DOWN
        dir_l = game.direction == Direction.LEFT
        dir_u = game.direction == Direction.UP

        state = [
            # danger straight
            (dir_r and game.is_collision(point_r))
            or (dir_d and game.is_collision(point_d))
            or (dir_l and game.is_collision(point_l))
            or (dir_u and game.is_collision(point_u)),
            # danger right
            (dir_r and game.is_collision(point_d))
            or (dir_d and game.is_collision(point_l))
            or (dir_l and game.is_collision(point_u))
            or (dir_u and game.is_collision(point_r)),
            # danger left
            (dir_r and game.is_collision(point_u))
            or (dir_d and game.is_collision(point_r))
            or (dir_l and game.is_collision(point_d))
            or (dir_u and game.is_collision(point_l)),
            # Move direction
            dir_r,
            dir_d,
            dir_l,
            dir_u,
            # Food position
            game.food.x < head.x,  # Food to the left
            game.food.x > head.x,  # Food to the right
            game.food.y < head.y,  # Food above
            game.food.y > head.y,  # Food below
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(
                self.memory, BATCH_SIZE
            )  # Returns a list of tuples of size BATCH_SIZE
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(
            *mini_sample
        )  # The * operator unpacks the tuples into individual lists which are then mapped to the LHS.

        # self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Random moves: tradeoff between exploration(random move) and exploitation(model prediction)

        self.epsilon = 80 - self.n_games  # 80 is hard coded.
        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
            # The more games we play, the smaller epsilon gets and we favour exploitation over exploration.
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGame()

    while True:
        # get current state
        state_old = agent.get_state(game)

        # Get move based on current state:
        final_move = agent.get_action(state_old)

        # Perform move and get new state:
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Remember:
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Train the long memory or replay memory.

            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                reward = 10
                agent.model.save()

            print(f"Game: {agent.n_games}, Score: {score}, Record: {record}")

            plot_scores.append(score)
            total_score += score
            mean_scores = total_score / agent.n_games
            plot_mean_scores.append(mean_scores)
            plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    train()
