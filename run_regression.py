import argparse
from regression_example.main import main

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='CPAB Activation')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
    
    args = parser.parse_args()
    
    main(args)