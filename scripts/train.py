import argparse
import yaml
import torch

from src.models.depthnet import SparseDepthNet
from src.sampler.policy import AdaptiveSamplingPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/base.yaml')
    args = parser.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    model = SparseDepthNet(cfg['model']['in_channels'])
    policy = AdaptiveSamplingPolicy()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(policy.parameters()),
                                 lr=cfg['train']['lr'])
    print('training for', cfg['train']['epochs'], 'epochs')


if __name__ == '__main__':
    main()
