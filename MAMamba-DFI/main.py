import os
import cv2
import random
import numpy as np
import torch
import argparse
from shutil import copyfile
from src.config import Config
from src.MAMamba-DFI import MAMamba-DFI
import wandb

def main(mode=None):
    r"""starts the model

    Args:
        mode (int): 1: train, 2: test, reads from config file if not specified
    """
    config = load_config(mode)
    with wandb.init(project='Rstormer', config=load_config(mode)):


        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)



        if torch.cuda.is_available():
            print('Cuda is available')
            config.DEVICE = torch.device("cuda")
            torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
        else:
            print('Cuda is unavailable, use cpu')
            config.DEVICE = torch.device("cpu")




        cv2.setNumThreads(0)



        torch.manual_seed(config.SEED)
        torch.cuda.manual_seed_all(config.SEED)
        np.random.seed(config.SEED)
        random.seed(config.SEED)




        model = MAMamba-DFI(config)
        model.load()



        if config.MODE == 1:
            config.print()
            print('\nstart training...\n')
            model.train()


        elif config.MODE == 2:
            print('\nstart testing...\n')
            model.test()




def load_config(mode=None):
    r"""loads model config

    Args:
        mode (int): 1: train, 2: test, reads from config file if not specified
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '--checkpoints', type=str, default='',
                        help='model checkpoints path (default: ./checkpoints)')

    parser.add_argument('--model', type=int, default='2', choices=[2])


    if mode == 2:
        parser.add_argument('--input', type=str,

                            default='',
                            help='path to the input images directory or an input image')


        parser.add_argument('--mask', type=str,

                            default='',
                            help='path to the masks directory or a mask file')


        parser.add_argument('--output', type=str,

                            default='',
                            help='path to the output directory')

    args = parser.parse_args()
    config_path = os.path.join(args.path, 'config.yml')



    if not os.path.exists(args.path):
        os.makedirs(args.path)


    if not os.path.exists(config_path):
        copyfile('./config.yml.example', config_path)


    config = Config(config_path)
    print(config_path)


    if mode == 1:
        config.MODE = 1
        if args.model:
            config.MODEL = args.model


    elif mode == 2:
        config.MODE = 2
        config.MODEL = args.model if args.model is not None else 3

        if args.input is not None:
            config.TEST_INPAINT_IMAGE_FLIST = args.input

        if args.mask is not None:
            config.TEST_MASK_FLIST = args.mask

        if args.output is not None:
            config.RESULTS = args.output


    return config


if __name__ == "__main__":
    main()
