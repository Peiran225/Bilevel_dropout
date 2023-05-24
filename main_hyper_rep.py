import os
import time
import pdb
import logging
import json
import argparse
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models

from utils_hyper import *
from dataset import *

import pdb

def finetunning(args, model, x_spt, y_spt, x_qry, y_qry):
    """
    :param x_spt:   [setsz, c_, h, w]
    :param y_spt:   [setsz]
    :param x_qry:   [querysz, c_, h, w]
    :param y_qry:   [querysz]
    :return:
    """
    assert len(x_spt.shape) == 4

    querysz = x_qry.size(0)

    losses_q = [0 for _ in range(args.innerT + 1)] 
    corrects = [0 for _ in range(args.innerT + 1)]


    # in order to not ruin the state of running_mean/variance and bn_weight/bias
    # we finetunning on the copied model instead of self.net
    net = deepcopy(model)
    # pdb.set_trace()
    # 1. run the i-th task and compute loss for k=0
    logits = net(x_spt)
    loss = F.cross_entropy(logits, y_spt)
    grad = torch.autograd.grad(loss, net.getInner_params())
    fast_weights = list(map(lambda p: p[1] - args.lr * p[0], zip(grad, net.getInner_params())))

    # this is the loss and accuracy before first update
    with torch.no_grad():
        # [setsz, nway]
        logits_q = net(x_qry, net.parameters(), bn_training=True)
        loss_q = F.cross_entropy(logits_q, y_qry)
        losses_q[0] += loss_q
        # [setsz]
        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
        correct = torch.eq(pred_q, y_qry).sum().item()
        corrects[0] = corrects[0] + correct
        

    #this is the loss and accuracy after the first update
    with torch.no_grad():
        # [setsz, nway]
        logits_q = net(x_qry, fast_weights, bn_training=True)
        loss_q = F.cross_entropy(logits_q, y_qry)
        losses_q[1] += loss_q
        ## [setsz]
        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
        ## scalar
        correct = torch.eq(pred_q, y_qry).sum().item()
        corrects[1] = corrects[1] + correct
        losses_q[1] = losses_q[1] + loss_q 

    for k in range(1, args.innerT):
    #    # 1. run the i-th task and compute loss for k=1~K-1
        logits = net(x_spt, fast_weights, bn_training=True)
        loss = F.cross_entropy(logits, y_spt)
    #   # 2. compute grad on theta_pi
        grad = torch.autograd.grad(loss, fast_weights)
    #    # 3. theta_pi = theta_pi - train_lr * grad
        fast_weights = list(map(lambda p: p[1] - args.lr * p[0], zip(grad, fast_weights)))

        logits_q = net(x_qry, fast_weights, bn_training=True)
        # loss_q will be overwritten and just keep the loss_q on last update step.
        loss_q = F.cross_entropy(logits_q, y_qry)
        losses_q[k + 1] += loss_q

        with torch.no_grad():
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
            corrects[k + 1] = corrects[k + 1] + correct


    del net

    accs = np.array(corrects) / querysz
    losses = np.array([l.data.cpu().numpy().item() for l in losses_q])

    return accs[-1], losses[-1]

def train(args, data_loader, logger):     

    eps = 1e-6

    config = [
        ('dropout1', [args.drop1]), 
        ('conv2d', [64, 1, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('dropout2', [args.drop2]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('dropout3', [args.drop3]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('dropout4', [args.drop4]),
        ('conv2d', [64, 64, 2, 2, 1, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('flatten', []),
        ('dropout5', [args.drop5]),
        ('linear', [args.n_way, 64])
    ]

    #inner variable
    model = Learner(config,args).cuda()
    model_old = Learner(config,args).cuda()
    assert len(model.parameters()) == 18

    for p, p_old in zip(model.parameters(), model_old.parameters()):
        p_old.data = p.data

    hyperDim = 0
    for p in model.getHyperRep_params():
        hyperDim += len(p.reshape(-1))
    
    innerDim = 0
    for p in model.getInner_params():
        innerDim += len(p.reshape(-1))

    print(hyperDim, innerDim)

    if args.outer_opt == 'SGD':
        opt_lamda = optim.SGD(model.getHyperRep_params(), lr=args.hlr)
    elif args.outer_opt == 'Adam':
        opt_lamda = optim.Adam(model.getHyperRep_params(), lr=args.hlr)

    ###########################Main training loop#################################################
    hyt = 0
    # pdb.set_trace()
    while True:
        for batch in data_loader.dataloader:
            x_spt, y_spt = batch['train']
            x_qry, y_qry = batch['test']
            x_spt, y_spt, x_qry, y_qry = x_spt.cuda(), y_spt.cuda(), x_qry.cuda(), y_qry.cuda()

            start_time = time.time()
            if  args.alg == 'FSLA': 
                if hyt == 0:
                    v_state = []
                    for p in model.getInner_params():
                        v_state.append(torch.zeros_like(p).cuda())

                tmp_grad = [torch.zeros_like(p).cuda() for p in model.getHyperRep_params()]
                tmp_v_state = [torch.zeros_like(p).cuda() for p in model.getInner_params()]
                tmp_v_norm = [0 for _ in np.arange(len(model.getInner_params()))]
                tmp_loss = []

                for t_num in range(args.task_num):
                    model.dropout1.p = 0
                    model.dropout2.p = 0
                    model.dropout3.p = 0
                    model.dropout4.p = 0
                    new_params = inner_update(args, model, x_spt, y_spt, t_num)
                    model.dropout1.p = args.drop1
                    model.dropout2.p = args.drop2
                    model.dropout3.p = args.drop3
                    model.dropout4.p = args.drop4
                    
                    model.dropout5.p = 0
                    loss, grad, v_s, v_norm = hyper_grad_fsla(args, model, new_params, x_spt, y_spt, x_qry, y_qry, t_num, v_state, hyt)
                    model.dropout5.p = args.drop5
                    
                    tmp_grad = [tmp_g + fast_g.detach().clone()/args.task_num for tmp_g, fast_g in zip(tmp_grad, grad)]
                    tmp_v_state = [tmp_g + fast_g.detach().clone()/args.task_num for tmp_g, fast_g in zip(tmp_v_state, v_s)]
                    tmp_v_norm = [tmp_g + fast_g/args.task_num for tmp_g, fast_g in zip(tmp_v_norm, v_norm)]
                    tmp_loss.append(loss)

                v_state = tmp_v_state

                if hyt % args.interval  == 0:
                    logger.update_err(np.mean(tmp_loss))
                    logger.update_v_norm(tmp_v_norm)
    

                opt_lamda.zero_grad()
                for p, g in zip(model.getHyperRep_params(), tmp_grad):
                    p.grad = g.detach().clone()
                opt_lamda.step()

            
            

            ######################################################################################################
            training_time = time.time() - start_time
            
            if hyt % args.interval == 0:
                logger.update_time(training_time)
                model.eval()

                ##################train acc##################
                train_accs = []; train_losses = []
                train_step = 0                
                # split to single task each time
                for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                    train_acc, train_loss = finetunning(args, model, x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                    train_accs.append(train_acc); train_losses.append(train_loss)
                    
                train_step += args.task_num
                if train_step > 100:
                      break

                logger.update_trainAcc(np.mean(train_accs))
                logger.save()
                
                ##################test acc################
                accs = []; losses = []
                test_step = 0                
                for test_batch in data_loader.dataloader_val:
                    x_spt, y_spt = test_batch['train']
                    x_qry, y_qry = test_batch['test']
                    x_spt, y_spt, x_qry, y_qry = x_spt.cuda(), y_spt.cuda(), x_qry.cuda(), y_qry.cuda()

                    # split to single task each time
                    for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                        test_acc, test_loss = finetunning(args, model, x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                        accs.append(test_acc); losses.append(test_loss)
                    
                    test_step += args.task_num
                    if test_step > 100:
                        break


                logger.update_testAcc(np.mean(accs))
                logger.print(hyt)
                logger.save()
                model.train()

            hyt += 1
            if hyt >= args.T:
                return

if __name__ == "__main__":
    # sending arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, default='FSLA', choices=['stocBiO', 'HOAG', 'TTSA', 'BSA',  'ESJ', 'HOZOJ', 'BiAdam', 'VR-BiAdam',
                                                        'reverse', 'AID_CG', 'AID_NS', 'VRBO', 'MRBO', 'MSTSA', 'Dire',\
                                                            'STABLE', 'AsBio', 'FSLA', 'FSLA_ADA', 'SMB', 'SVRB', 'HFBiO_vanilla', 'HFBiO', 'HFBiO_special'])
    parser.add_argument('--data', type=str, default='Omniglot', choices=['Omniglot', 'MiniImageNet'])                    
    parser.add_argument('--outer_opt', type=str, default='SGD', choices=['Adam', 'SGD'])
    parser.add_argument('--innerT', type=int, default= 4, help="Number of Inner Iters")
    parser.add_argument('--T', type=int, default=10000, help="Number of Outer Iters")
    parser.add_argument('--n_way', type=int, help='number classes', default=5)
    parser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    parser.add_argument('--k_qry', type=int, help='number samples for query set', default=15)
    parser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    parser.add_argument('--imgc', type=int, help='imgc', default=1)
    parser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    parser.add_argument('--v_iter', type=int, default=1, help="Number of iterations to compute v")
    parser.add_argument('--spider_iters', type=int, default=3, help="Spider Frequency")
    parser.add_argument('--spider_batch_size', type=int, default=0, help="Spider Batch_size")
    parser.add_argument('--hlr', type=float, default=0.1, help="HyperLr")#0.1
    parser.add_argument('--lr', type=float, default= 0.8, help="InnerLr")#0.4
    parser.add_argument('--beta', type=float, default= 0.05, help="Shrinkage parameter used in Neumann series")#0.5
    parser.add_argument('--storm_coef', type=float, default= 1, help="")
    parser.add_argument('--beta_adam', type=float, default= 0.99, help="Exponetial Moving Average Coefficient")
    parser.add_argument('--lamda', type=float, default= 0.01, help="Regularization parameter for v using conjugate")
    parser.add_argument('--interval', type=int, help='', default=500)
    parser.add_argument('--drop1', type=float, default = 0.3, help="Dropout OR NOT")
    parser.add_argument('--drop2', type=float, default = 0.3, help="Dropout OR NOT")
    parser.add_argument('--drop3', type=float, default = 0.3, help="Dropout OR NOT")
    parser.add_argument('--drop4', type=float, default = 0.3, help="Dropout OR NOT")
    parser.add_argument('--drop5', type=float, default = 0, help="Dropout OR NOT")


    parser.add_argument('--v_beta1', type=float, default= 0.9, help="Exponetial Moving Average Coefficient")
    parser.add_argument('--v_beta2', type=float, default= 0.99, help="Exponetial Moving Average Coefficient")

    args = parser.parse_args()
    config = args

    prefix_dir =  '.'
    if args.data == 'Omniglot':
        data_loader = OmniglotNShot(batchsz=args.task_num, n_way = args.n_way, k_shot=args.k_spt, k_query=args.k_qry, spider_batchsz=args.spider_batch_size)
    elif args.data == 'MiniImageNet':
        data_loader = MiniImagenetNShot(batchsz=args.task_num, n_way = args.n_way, k_shot=args.k_spt, k_query=args.k_qry)
    prefix = str(args.n_way) + '-' + str(args.k_spt) + '-' + str(args.k_qry) + '-' + str(args.task_num) + '-' + args.alg + '-' + args.outer_opt

    postfix = 'T-' + str(args.T) + '-hlr-' + str(args.hlr) + '-innerT-' + str(args.innerT) + '-lr-' + str(args.lr) + '-beta-' + str(args.beta) + '-drop1-' + str(args.drop1) + '-drop5-' + str(args.drop5)
    print(postfix)
    if  args.alg == 'FSLA':
        postfix +=  '-beta-' + str(args.beta) + '-v_iter-' + str(args.v_iter)

    
    logger = Logger_meta(prefix_dir + '/hyper_rep', prefix = prefix, postfix= postfix)

    train(config, data_loader, logger)
