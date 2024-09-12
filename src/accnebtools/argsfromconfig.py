import yaml
import argparse
from pydoc import locate


def make_parser(method_config_yaml, name):
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file", type=str,
                        help="File containing the graph as .npz file.",
                        )
    parser.add_argument("output_file", type=str,
                        help="Where embeddings will be saved. Numpy .npy files are used.",
                        )
    parser.add_argument("--metadata", type=str, default=None,
                        help="Path where embedding method metadata will be saved.")
    parser.add_argument("--config", type=str, default=None,
                        help="Json file containing all necessary configs.")
    parser.add_argument('--undirected', type=int, default=0,
                        help='Do not differentiate between in and out degrees.')
    parser.add_argument('--weighted', type=int, default=0, help='Use weights.')
    parser.add_argument('--node_attributed', type=int, default=0, help='Use node attributes.')
    parser.add_argument("--timeout", type=int, default=1000,
                        help="Time before method is timed out.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed to use for model.")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Currently not implemented for all algorithms.")
    with open(method_config_yaml, 'r') as fp:
        config = yaml.safe_load(fp)

    hyperparams = config['methods'][name]['hyperparameters']
    if hyperparams is not None:
        for param_name, values in hyperparams.items():
            if values['type'] in ['int', 'str', 'float']:
                parser.add_argument("--" + str(param_name),
                                    type=locate(values['type']),
                                    default=values['default'],
                                    help=values['description'])
            elif values['type'] == 'bool':
                parser.add_argument("--" + str(param_name),
                                    type=int,
                                    default=int(values['default']),
                                    help=values['description'])
            elif values['type'] == 'tuple':
                parser.add_argument("--" + str(param_name),
                                    type=tuple,
                                    nargs=values['nargs'],
                                    default=values['default'],
                                    help=values['description'])
            elif values['type'] == 'list':
                parser.add_argument("--" + str(param_name),
                                    type=list,
                                    nargs=values['nargs'],
                                    default=values['default'],
                                    help=values['description'])

    return parser
