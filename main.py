from multiprocessing import reduction
import os
import time
import pdb
import logging
import json
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models

from utils import *

import pdb

def train(args, data_loader, logger):     

    eps = 1e-6

    #inner variable
    #config = [
    #     ('linear', [args.num_class, 28*28]),
    #]

    config = [
         ('dropout', [args.drop]),
         ('linear', [1024, 28*28]),
         ('relu', [True]),
         ('dropout', [args.drop]),
         ('linear', [1024, 1024]),
         ('relu', [True]),
         ('dropout', [args.drop]),
         ('linear', [2048, 1024]),
         ('relu', [True]),
         ('dropout', [args.drop]),
         ('linear', [args.num_class, 2048]),
    ]

    #config = [
    #     ('dropout', [0]),
    #     ('linear', [20*20*3, 32*32*3]),#originally 16*16, 28*28
    #     ('relu', [True]),
    #     ('dropout', [0]),
    #     ('linear', [10*10*3, 20*20*3]),#orinally 8*8, 16*16
    #     ('relu', [True]),
    #     ('dropout', [0]),
    #     ('linear', [args.num_class, 10*10*3]),#orinally 8*8
    #]
    #config = [
    #    ('conv2d', [64, 3, 3, 3, 2, 0]),
    #    ('relu', [True]),
    #    ('bn', [64]),
    #    ('dropout', [0]),
    #    ('conv2d', [64, 64, 3, 3, 2, 0]),
    #    ('relu', [True]),
    #    ('bn', [64]),
    #    ('dropout', [0]),
    #    ('conv2d', [64, 64, 3, 3, 2, 0]),
    #    ('relu', [True]),
    #    ('bn', [64]),
    #    ('dropout', [0]),
    #    ('conv2d', [64, 64, 2, 2, 1, 0]),
    #    ('relu', [True]),
    #    ('bn', [64]),
    #    ('flatten', []),
    #    ('dropout', [0]),
    #    ('linear', [10, 64*4]),
    #    ('relu', [True]),
    #    ('dropout', [0]),
    #    ('linear', [10, 10])
    #]


    inner = Learner(config).cuda()
    inner_old = Learner(config).cuda()

    for p, p_old in zip(inner.parameters(), inner_old.parameters()):
        p_old.data = p.data
    # pdb.set_trace()
    ld = torch.tensor(np.random.normal(size=args.num_sample), dtype=torch.float).cuda().requires_grad_()
    ld_old = torch.zeros_like(ld).requires_grad_()
    ld_old.data = ld.data

    if args.cr == 'CE':
        args.crit = nn.CrossEntropyLoss(reduction='none')
        args.crit_mean = nn.CrossEntropyLoss()
    elif args.cr == 'SH':
        args.crit = nn.MultiMarginLoss(p=2, reduction='none')
        args.crit_mean = nn.MultiMarginLoss(p=2)


    if args.outer_opt == 'SGD':
        opt_lamda = optim.SGD([ld,], lr=args.hlr)
    elif args.outer_opt == 'Adam':
        opt_lamda = optim.Adam([ld,], lr=args.hlr)

    ###########################Main training loop#################################################
    for hyt in range(args.T):
        start_time = time.time()
        

        if args.alg == 'FSLA': 
            if hyt == 0:
                v_state = []
                for p in inner.parameters():
                    v_state.append(torch.zeros_like(p).cuda())
            inner.train()
            inner = inner_update2(args, args.batch_size, inner, ld, data_loader)
            inner.eval()
            loss, grad_lamda, v_state, v_norm = hyper_grad_fsla(args, args.batch_size, inner, ld, data_loader, v_state)
            logger.update_err(loss)

            opt_lamda.zero_grad()
            ld.grad = grad_lamda.detach().clone()
            opt_lamda.step()

            logger.update_gnorm(torch.norm(grad_lamda).data.cpu().numpy().item()) 
            logger.update_v_norm(v_norm)

        
                     
                     
            
        
        ###################################################################################################### 
        training_time = time.time() - start_time
        logger.update_time(training_time)                
        logger.update_f(compute_f1_score(ld, data_loader.y_index))
        # print(ld[:50])
        
        with torch.no_grad():
            inner.eval()
            #########test acc###########
            testx, testy = data_loader.get_test() #CNN
            tx = torch.tensor(testx, dtype=torch.float).reshape(-1, args.size**2*args.channel).cuda() #linear
            #tx = torch.tensor(testx, dtype=torch.float).reshape(-1, args.channel, args.size, args.size).cuda() #CNN

            #ans = torch.argmax(inner(testx),-1)   #CNN
            ans = torch.argmax(inner(tx),-1) #linear
            correct = ans == testy
            acc = torch.mean(correct.float())
            logger.update_testAcc(acc.data.cpu().numpy().item())
            

            #########train acc###########
            trainx, trainy = data_loader.get_train() #CNN 
            tx = torch.tensor(trainx, dtype=torch.float).reshape(-1, args.size**2*args.channel).cuda() #linear
            #tx = torch.tensor(trainx, dtype=torch.float).reshape(-1, args.channel, args.size, args.size).cuda() #CNN
            #ans = torch.argmax(inner(testx),-1)   #CNN
            ans = torch.argmax(inner(tx),-1) #linear
            correct = ans == trainy
            train = torch.mean(correct.float())
            logger.update_trainAcc(train.data.cpu().numpy().item())

            inner.train




            
        if hyt %  100==0: 
            logger.print(hyt)
            logger.save()

if __name__ == "__main__":
    # sending arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, default='FSLA', choices=['FSLA', 'FSLAdrop'])
    parser.add_argument('--outer_opt', type=str, default='SGD', choices=['Adam', 'SGD'])
    parser.add_argument('--cr', type=str, default='CE', choices=['CE', 'SH'])
    parser.add_argument('--innerT', type=int, default= 1, help="Number of Inner Iters")
    parser.add_argument('--T', type=int, default=2000, help="Number of Outer Iters")
    parser.add_argument('--v_iter', type=int, default=3, help="Number of iterations to compute v")
    parser.add_argument('--spider_iters', type=int, default=3, help="Spider Frequency")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch_size")
    parser.add_argument('--spider_batch_size', type=int, default=5000, help="Spider Batch_size")
    parser.add_argument('--num_class', type=int, default=10, help="Number Class")

    parser.add_argument('--hlr', type=float, default=1, help="HyperLr")#no dropout defaut: 1, with dropout defaut: 1
    parser.add_argument('--lr', type=float, default= 0.01, help="InnerLr")#no dropout defaut: 0.01, with dropout defaut: 0.01
    parser.add_argument('--gamma', type=float, default= 0.95, help="momentum")
    parser.add_argument('--beta', type=float, default= 0.025, help="Shrinkage parameter used in Neumann series")#no dropout defaut 0.05, with dropout defaut: 0.05
    parser.add_argument('--storm_coef', type=float, default= 1, help="Shrinkage parameter used in Neumann series")
    parser.add_argument('--beta_adam', type=float, default= 0.99, help="Exponetial Moving Average Coefficient")
    parser.add_argument('--lamda', type=float, default= 0.01, help="Regularization parameter for v using conjugate")
    parser.add_argument('--drop', type = str, default = 'TRUE', help="Dropout OR NOT")
    parser.add_argument('--max', type=float, default= 3, help="Max-norm Regularization")
    parser.add_argument('--sample', type=int, default= 40000, help="Max-norm Regularization")

    #L1
    parser.add_argument('--th', type=float, default=0, help="")
    parser.add_argument('--l1_alpha', type=float, default=0, help="")
    parser.add_argument('--lasso', dest='lasso', action='store_true', help='')


    parser.add_argument('--rho', type=float, default= 0.6, help="Noise Rate")
    parser.add_argument('--channel', type=int, default=1, help="Batch_size")
    parser.add_argument('--size', type=int, default=28, help="Batch_size")
    parser.add_argument('--num_sample', type=int, default=40000, help="Batch_size")

    

    args = parser.parse_args()
    args.T = 10000
    for args.drop in [0.1,0.2,0.3,0.4,0.5]:
        args.sample = '5000'

        print(args.lasso)
    
        prefix_dir ='.' #'/ocean/projects/cis220038p/junyili/AdaBilevel'
    
        data_loader = Data('./preprocessed_data/processed_data/MNIST_data_5000/', args.rho, args.size, args.channel)
        # data_loader = Data('./SVHN_data/')
        #data_loader = Data('./FMNIST_data/')
        #data_loader = Data('./preprocessed_data/processed_data/CIFAR10_data/', args.rho, args.size, args.channel)
        # data_loader = Data('./QMNIST_data/')

        prefix = 'with_train_acc' + args.alg + '-' + args.outer_opt

        postfix = 'rho-' + str(args.rho) + '-T-' + str(args.T) + '-hlr-' + str(args.hlr) + '-lr-' + str(args.lr) + '-Sample-' + str(args.sample) + '-DropoutRate-' + str(args.drop)
        if args.alg == 'AID_CG':
            postfix += '-v_iter-' + str(args.v_iter) + '-lamda-' + str(args.lamda) + '-beta-' + str(args.beta)
        elif args.alg == 'AID_NS':
            postfix += '-v_iter-' + str(args.v_iter) + '-beta-' + str(args.beta)
        elif args.alg == 'VRBO':
            postfix += '-spider_iters-' + str(args.spider_iters) + '-spider_bs-' + str(args.spider_batch_size)
        elif args.alg == 'AsBio':
            postfix += '-spider_iters-' + str(args.spider_iters) + '-spider_bs-' + str(args.spider_batch_size) + '-beta_adam-' + str(args.beta_adam) + '-s_coef-' + str(args.storm_coef)
        elif args.alg == 'MRBO':
            postfix += '-d-' + str(args.d) + '-m-' + str(args.m) + '-c_lamda-' + str(args.c_lamda) + '-c_inner-' + str(args.c_inner)
        elif args.alg == 'MSTSA':
            postfix +=  '-c_lamda-' + str(args.c_lamda)
        elif args.alg == 'SMB':
            postfix += '-c_lamda-' + str(args.c_lamda) + '-c_inner-' + str(args.c_inner)
        elif args.alg == 'HFBiO_vanilla':
            postfix += '-tau-' + str(args.tau) + '-' + args.cr
        elif args.alg == 'HFBiO_special':
            postfix += '-tau-' + str(args.tau) + '-mu-' + str(args.mu) + '-niu-' + str(args.niu)+  '-' + args.cr
        elif args.alg == 'FSLA':
            postfix += '-beta-' + str(args.beta)
        elif args.alg == 'FSLA_ADA':
            postfix += '-beta-' + str(args.beta) + '-beta1-' + str(args.v_beta1) + '-beta2-' + str(args.v_beta2)

    
        if args.lasso:
            postfix += '-th-' + str(args.th)

        if args.l1_alpha > 0:
            postfix += '-alpha-' + str(args.l1_alpha)
    
        print(postfix)
        logger = Logger(prefix_dir + '/data_cleaning_new2', prefix = prefix, postfix= postfix)

    
        train(args, data_loader, logger)
