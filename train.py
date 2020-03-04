import argparse

from car_detection.model import VehicleModel


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True,
                        help="name of pickle file")
    parser.add_argument("-c", "--create", required=True,
                        help="create pickle file", type=bool)
    parser.add_argument("-e", "--epoch", required=True,
                        help="Number of epoch for train")

    return parser.parse_args()


if __name__ == "__main__":

    args = _parse_args()

    VehicleModel(args.name).train(create=args.create, epoch_count=int(args.epoch))
