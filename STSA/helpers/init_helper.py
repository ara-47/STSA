import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_logger(log_dir: str, log_file: str) -> None:
    logger = logging.getLogger()
    format_str = r'[%(asctime)s] %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        datefmt=r'%Y/%m/%d %H:%M:%S',
        format=format_str
    )
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(log_dir / log_file))
    fh.setFormatter(logging.Formatter(format_str))
    logger.addHandler(fh)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # model type
    parser.add_argument('model', type=str, default='STVT')

    # training & evaluation
    parser.add_argument('--device', type=str, default='cuda',
                        choices=('cuda', 'cpu'))
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--splits', type=str, nargs='+', default=[])
    parser.add_argument('--max-epoch', type=int, default=300)
    parser.add_argument('--model-dir', type=str, default='../models/model')
    parser.add_argument('--log-file', type=str, default='log.txt')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--lambda-reg', type=float, default=1.0)
    parser.add_argument('--nms-thresh', type=float, default=0.5)

    # inference
    parser.add_argument('--ckpt-path', type=str, default=None)
    parser.add_argument('--sample-rate', type=int, default=15)
    parser.add_argument('--source', type=str, default=None)
    parser.add_argument('--save-path', type=str, default=None)

    # common model config
    parser.add_argument('--dataset', type=str, default='SumMe')
    parser.add_argument('--head_dim', type=int, default=64)
    parser.add_argument('--num_channels', type=int, default=512)
    parser.add_argument('--num_slots', type=int, default=10)
    parser.add_argument('--embedding_dim', type=int, default=768)
    parser.add_argument('--num_iters', type=int, default=3)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--hidden_dim', type=int, default=3072)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--attn_dropout_rate', type=float, default=0.0)
    parser.add_argument('--use_representation', type=bool, default=True)
    parser.add_argument('--conv_patch_representation', type=bool, default=False)
    parser.add_argument('--positional_encoding_type', type=str, default='learned')
    


    

    return parser


def get_arguments() -> argparse.Namespace:
    parser = get_parser()
    args = parser.parse_args()
    return args
