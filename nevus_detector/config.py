import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Training and validating a nevus detector model.")

    parser.add_argument('--main_dir', default='/research/labs/ophthalmology/iezzi/m294666/', type=str,
                        help="The main directory")

    # main_dir = 'C:/Users/m294666/Documents/'
    # main_dir = '/research/labs/ophthalmology/iezzi/m294666/'

    parser.add_argument(
        '--pretrained_dir',
        default='/research/labs/ophthalmology/iezzi/m294666/pretrained_models/resnet_pretrained_weights_distorted.pth',
        type=str,
        help="The directory for the pretrained model")

    parser.add_argument('--batch_size', default=[64],
                        help="Batch size (default: 32)")

    parser.add_argument('--num_epochs', default=100, type=int,
                        help="Number of epochs (default: 1)")

    parser.add_argument('--learning_rate', default=[0.001],
                        help="Learning rate (default: 0.001)")

    args = parser.parse_args()
    return args