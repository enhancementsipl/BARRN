import argparse, time, os
import imageio

import options.options as option
from utils import util
from solvers import create_solver
from data import create_dataloader
from data import create_dataset

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

def main():
    parser = argparse.ArgumentParser(description='Test Super Resolution Models')
    parser.add_argument('-opt', default='./options/test/test_SRFBN_AR.json',type=str,  help='Path to options JSON file.')
    opt = option.parse(parser.parse_args().opt)
    opt = option.dict_to_nonedict(opt)

    solver = create_solver(opt)


    solver.feed_data()

    # calculate forward time
    t0 = time.time()
    solver.test()
    t1 = time.time()
    print(t1-t0)

if __name__ == '__main__':
    main()