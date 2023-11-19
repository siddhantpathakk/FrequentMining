import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_path', type=str, default='data/task1', help='path to dataset csv', required=True)
    parser.add_argument('--output_path', type=str, default='output/task1', help='path to output csv', required=True)
    parser.add_argument('--support', type=int, default=3, help='support value', required=True)
    parser.add_argument('--debug', type=bool, default=False, help='debug mode', required=False)
    
    return parser.parse_known_args()