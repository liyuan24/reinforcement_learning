from collections import defaultdict
import random
from env import TicTacToeEnvironment


class Agent:
    def __init__(self, player_id, epsilon=0.1):
        self.player_id = player_id
        self.epsilon = epsilon
        self.value_function = defaultdict(float)

    def get_state_value(self, state):
        """Get the estimated value of a state"""
        return self.value_function[state]

    def choose_action(self, env, training=True):
        """Choose action using epsilon-greedy policy"""
        available_actions = env.get_available_actions()

        if not available_actions:
            return None

        if training and random.random() < self.epsilon:
            # Explore: random action
            return random.choice(available_actions)
        else:
            # Exploit: choose action leading to highest value state
            best_action = None
            best_value = float("-inf")

            current_player = env.current_player

            for action in available_actions:
                # Simulate the move
                temp_env = TicTacToeEnvironment()
                temp_env.board = env.board.copy()
                temp_env.current_player = current_player

                next_state, reward, done = temp_env.make_move(action)

                if done:
                    # Terminal state value is just the reward
                    value = reward
                else:
                    # Non-terminal state value from value function
                    value = -self.get_state_value(next_state)

                if value > best_value:
                    best_value = value
                    best_action = action

            return best_action


if __name__ == "__main__":
    env = TicTacToeEnvironment()
    agent = Agent(1)
    print(agent.choose_action(env))
