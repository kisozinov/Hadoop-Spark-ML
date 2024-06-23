from argparse import ArgumentParser, Namespace

def parse_arguments() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument('--num-nodes', '-n', choices=['1', '3'], default = '1')
    parser.add_argument('--optimize', '-o', action='store_true')
    return parser.parse_args()