from trainer import TicTacToeTrainer

def main():
    # Initialize trainer
    trainer = TicTacToeTrainer()
    
    # Train the agents
    print("Training tic-tac-toe agents using value function estimation...")
    stats = trainer.train(episodes=20000)
    
    # Save the trained agents
    trainer.save_agents()
    
    # Play against human
    while True:
        play_again = input("\nWould you like to play against the AI? (y/n): ").lower()
        if play_again != 'y':
            break
            
        player_choice = input("Do you want to be X or O? (X goes first): ").upper()
        if player_choice not in ['X', 'O']:
            player_choice = 'X'
            
        trainer.play_human_game(human_player=player_choice)
    
    print("Thanks for playing!")


if __name__ == "__main__":
    main()