import argparse
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Advanced Battery Failure Predictor and Management System",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        'command', 
        help="""
The command to execute. Choose from:
  train          - Train the multimodal fault prediction model.
  monitor        - Start the real-time fault monitoring system.
  train-reinforce- Train the basic RL agent for charging optimization.
  train-ppo      - Train the advanced PPO agent for charging optimization.
  generate-data  - Generate the dummy CSV data files.
"""
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args([sys.argv[1]])

    if args.command == 'train':
        print("Starting multimodal model training...")
        from train_multimodal import main as train_main
        train_main()
    elif args.command == 'monitor':
        print("Starting real-time monitoring system...")
        from realtime_monitoring import run_monitoring_system
        run_monitoring_system()
    elif args.command == 'train-reinforce':
        print("Starting REINFORCE training...")
        from train_reinforce_agent import main as reinforce_main
        reinforce_main()
    elif args.command == 'train-ppo':
        print("Starting PPO training...")
        from train_ppo_agent import main as ppo_main
        ppo_main()
    elif args.command == 'generate-data':
        print("Generating dummy data...")
        from generate_dummy_data import main as data_main
        data_main()
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help(sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
