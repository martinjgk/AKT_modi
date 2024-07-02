import sys
import os
import os.path
import glob
import logging
import argparse
import numpy as np
import torch
from load_data import DATA, PID_DATA
from run import train, test
from utils import try_makedirs, load_model, get_file_name_identifier
from main import test_one_dataset, test_one_datum
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Script to test KT')
    # Basic Parameters
    parser.add_argument('--max_iter', type=int, default=300,
                        help='number of iterations')
    parser.add_argument('--train_set', type=int, default=1)
    parser.add_argument('--seed', type=int, default=224, help='default seed')

    # Common parameters
    parser.add_argument('--optim', type=str, default='adam',
                        help='Default Optimizer')
    parser.add_argument('--batch_size', type=int,
                        default=24, help='the batch size')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate')
    parser.add_argument('--maxgradnorm', type=float,
                        default=-1, help='maximum gradient norm')
    parser.add_argument('--final_fc_dim', type=int, default=512,
                        help='hidden state dim for final fc layer')

    # AKT Specific Parameter
    parser.add_argument('--d_model', type=int, default=256,
                        help='Transformer d_model shape')
    parser.add_argument('--d_ff', type=int, default=1024,
                        help='Transformer d_ff shape')
    parser.add_argument('--dropout', type=float,
                        default=0.05, help='Dropout rate')
    parser.add_argument('--n_block', type=int, default=1,
                        help='number of blocks')
    parser.add_argument('--n_head', type=int, default=8,
                        help='number of heads in multihead attention')
    parser.add_argument('--kq_same', type=int, default=1)

    # AKT-R Specific Parameter
    parser.add_argument('--l2', type=float,
                        default=1e-5, help='l2 penalty for difficulty')

    # Datasets and Model
    parser.add_argument('--model', type=str, default='akt_pid',
                        help="combination of akt/sakt/dkvmn/dkt (mandatory), pid/cid (mandatory) separated by underscore '_'. For example tf_pid")
    parser.add_argument('--dataset', type=str, default="assist2009_pid")

    params = parser.parse_args()
    dataset = params.dataset

    if dataset in {"assist2009_pid"}:
        params.n_question = 110
        params.batch_size = 24
        params.seqlen = 200
        params.data_dir = 'data/'+dataset
        params.data_name = dataset
        params.n_pid = 16891

    if dataset in {"assist2017_pid"}:
        params.batch_size = 24
        params.seqlen = 200
        params.data_dir = 'data/'+dataset
        params.data_name = dataset
        params.n_question = 102
        params.n_pid = 3162

    if dataset in {"assist2015"}:
        params.n_question = 100
        params.batch_size = 24
        params.seqlen = 200
        params.data_dir = 'data/'+dataset
        params.data_name = dataset

    if dataset in {"statics"}:
        params.n_question = 1223
        params.batch_size = 24
        params.seqlen = 200
        params.data_dir = 'data/'+dataset
        params.data_name = dataset

    params.save = params.data_name
    params.load = params.data_name
    # Setup
    if "pid" not in params.data_name:
        dat = DATA(n_question=params.n_question,
                    seqlen=params.seqlen, separate_char=',')
    else:
        dat = PID_DATA(n_question=params.n_question,
                        seqlen=params.seqlen, separate_char=',')
    seedNum = params.seed
    np.random.seed(seedNum)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seedNum)
    np.random.seed(seedNum)
    file_name_identifier = get_file_name_identifier(params)

    file_name = ''
    for item_ in file_name_identifier:
        file_name = file_name+item_[0] + str(item_[1])

    test_data_path = params.data_dir + "/" + \
        params.data_name + "_test"+str(params.train_set)+".csv"
    test_q_data, test_qa_data, test_index = dat.load_data(
        test_data_path)

    print(test_q_data)
    print(test_qa_data)
    print(test_index)


def testStart():
    # test_one_datum(params, file_name, test_q_data, test_qa_data, test_index)
    test_one_dataset(params, file_name, test_q_data, test_qa_data, test_index)