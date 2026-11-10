import argparse
import torch

from src.models.depthnet import SparseDepthNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    model = SparseDepthNet()
    state = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(state)
    print('loaded checkpoint:', args.checkpoint)


if __name__ == '__main__':
    main()
