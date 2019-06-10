import os
import argparse
from solver import Solver
from data_loader import data_loader
from torch.backends import cudnn


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Data loader.
    
    dloader = data_loader(config.data_dir, batch_size=config.batch_size, mode=config.mode,num_workers=config.num_workers)

    # Solver for training and testing StarGAN.
    solver = Solver(dloader, config)

    if config.mode == 'train':
        solver.train()

    elif config.mode == 'test':
        solver.test()
      


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.

    parser.add_argument('--lambda_cycle', type=float, default=3, help='weight for cycle loss')
    parser.add_argument('--lambda_cls', type=float, default=2, help='weight for domain classification loss')
    
    parser.add_argument('--lambda_identity', type=float, default=2, help='weight for identity loss')
    
    # Training configuration.
    
    parser.add_argument('--batch_size', type=int, default=4, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--c_lr', type=float, default=0.0001, help='learning rate for C')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    
    

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')
    parser.add_argument('--src_speaker', type=str, default=None, help='test model source speaker')
    parser.add_argument('--trg_speaker', type=str, default="['SF1', 'TM1']", help='string list repre of target speakers eg."[a,b]"')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Directories.
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--test_dir', type=str, default='data/speakers_test')
    parser.add_argument('--log_dir', type=str, default='starganvc/logs')
    parser.add_argument('--model_save_dir', type=str, default='starganvc/models')
    parser.add_argument('--sample_dir', type=str, default='starganvc/samples')
    parser.add_argument('--result_dir', type=str, default='starganvc/results')
    
    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=2000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=100000)

    config = parser.parse_args()
    print(config)
    main(config)
    