from agent import Agent
from env import TicTacToeEnvironment
import pickle


class TicTacToeTrainer:
    def __init__(self, learning_rate=0.1, epsilon=0.1):
        self.self_play_agent = Agent(player_id=1, epsilon=epsilon)
        self.env = TicTacToeEnvironment()
        self.learning_rate = learning_rate

    def train(self, episodes=10000):
        """Train the agents through self-play"""

        for episode in range(episodes):
            done = False
            self.env.reset(current_player=1)

            while not done:
                current_state = self.env.get_state()

                action = self.self_play_agent.choose_action(self.env, training=True)
                if action is None:
                    break

                next_state, reward, done = self.env.make_move(action)
                # update value function with reward
                if done:
                    # Terminal state: TD target is just the reward
                    td_target = reward
                else:
                    # Non-terminal: TD target includes discounted next state value
                    td_target = reward - self.self_play_agent.get_state_value(
                        next_state
                    )

                # Update current agent's value function
                current_value = self.self_play_agent.get_state_value(current_state)
                td_error = td_target - current_value
                self.self_play_agent.value_function[current_state] += (
                    self.learning_rate * td_error
                )

            # Decay epsilon for exploration
            if episode % 1000 == 0:
                self.self_play_agent.epsilon *= 0.99
                self.learning_rate *= 0.99

        print(f"\nTraining completed!")

    def play_human_game(self, human_player="X"):
        """Play a game against human"""
        self.env.reset()

        if human_player == "X":
            human_id = 1
            ai_id = -1
        else:
            human_id = -1
            ai_id = 1

        print(f"\nYou are playing as {human_player}")
        print("Enter your move as row,col (e.g., 1,2)")

        while True:
            self.env.print_board()

            if self.env.current_player == human_id:
                # Human turn
                try:
                    move_input = input(f"\nYour move ({human_player}): ").strip()
                    row, col = map(int, move_input.split(","))
                    action = (row, col)

                    if action not in self.env.get_available_actions():
                        print("Invalid move! Try again.")
                        continue

                except (ValueError, IndexError):
                    print("Invalid input! Use format: row,col")
                    continue
            else:
                # AI turn
                action = self.self_play_agent.choose_action(self.env, training=False)
                print(f"\nAI plays: {action}")

            state, reward, done = self.env.make_move(action)

            if done:
                self.env.print_board()
                winner = self.env.check_winner()
                if winner is None:
                    print("It's a draw!")
                elif winner == human_id:
                    print(f"You win!")
                else:
                    print(f"AI wins!")
                break

    def save_agents(self, filename="tictactoe_agents.pkl"):
        """Save trained agents"""
        with open(filename, "wb") as f:
            pickle.dump(
                {
                    "self_play_agent": self.self_play_agent.value_function,
                },
                f,
            )
        print(f"Agents saved to {filename}")

    def load_agents(self, filename="tictactoe_agents.pkl"):
        """Load trained agents"""
        try:
            with open(filename, "rb") as f:
                data = pickle.load(f)
                self.self_play_agent.value_function = data["self_play_agent"]
            print(f"Agents loaded from {filename}")
        except FileNotFoundError:
            print(f"No saved agents found at {filename}")
