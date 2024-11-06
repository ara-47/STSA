
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import subprocess
import platform
import sys
import pkg_resources
import h5py
import json
import ortools
from tqdm import tqdm
import time
import datetime
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import tensorboardX
import numpy as np
from STSA.metrics import Metric
from STSA.build_dataloader import build_dataloader
from STSA.build_model import build_model
from STSA.build_optimizer import build_optimizer
from STSA.eval import select_keyshots
import pandas as pd

from STSA.utils import (
    adjust_learning_rate,
    save_model,
    load_model,
    resume_model,
)


pd_epoch = []
pd_batch_size = []
pd_lr = []
pd_runtime = []
pd_loss = []
pd_F_measure_k = []
best_fscore = 0

def parse_args():
    parser = argparse.ArgumentParser(prog='Spatio-Temporal Set Attention', description='Capturing Spatio-Temporal Dependencies with Competitive Set Attention for Video Summarization')
    parser.add_argument(
        '--exp_name', type=str, default='Experiment_', help='Experiments name'
    )
    parser.add_argument(
        '--dataset', default='TVSum', help='Dataset names.'
    )
    parser.add_argument(
        '--test_dataset',
        type=str,
        # the default case is random
        default="1,2,11,16,18,20,31,32,41,50",
        
        help='The number of test video in the dataset.',
        
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=2,
        help='The number of classes in the dataset.',
    )
    parser.add_argument(
        '--sequence',
        type=int,
        default=9,
        help='The number of frame sequence.',
    )
    parser.add_argument(
        '--num_sets',
        type=int,
        default = 10,
        help='The number of sets.',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=40,
        help='input batch size for training',
    )
    parser.add_argument(
        '--val_batch_size',
        type=int,
        default=40,
        help='input batch size for val',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default = 100,
        help='number of epochs to train'
    )
    parser.add_argument(
        '--test_epochs',
        type=int,
        default=1,
        help='number of internal epochs to test',
    )
    parser.add_argument('--optim', default='sgd', help='Model names.')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument(
        '--warmup_epochs',
        type=float,
        default=10,
        help='number of warmup epochs',
    )
    parser.add_argument(
        '--momentum', type=float, default=0.9, help='SGD momentum'
    )
    parser.add_argument(
        '--weight_decay', type=float, default=0.00008, help='weight decay'
    )
    parser.add_argument(
        '--nesterov',
        action='store_true',
        default=True,
        help='To use nesterov or not.',
    )
    parser.add_argument(
        '--no_cuda',
        action='store_true',
        default=False,
        help='disables CUDA training',
    )
    parser.add_argument(
        '--lr_scheduler',
        type=str,
        default="cosine",
        choices=["linear", "cosine"],
        help='how to schedule learning rate',
    )
    parser.add_argument(
        '--resume', action='store_true', default=False, help='Resume training'
    )
    parser.add_argument(
        '--gpu_id', default='1', type=str, help='id(s) for CUDA_VISIBLE_DEVICES'
    )


    parser.print_help()

    args = parser.parse_args()


    return args

def val(model, val_loader, epoch, args):

    global best_fscore
    
    model.eval()
    if epoch == -1:
        epoch = args.epochs - 1

    global pd_F_measure_k
    
    

    with tqdm(
        total=len(val_loader), desc='Validate Epoch #{}'.format(epoch + 1)
    ) as t:
        with torch.no_grad():
            predicted_multi_list = []
            target_multi_list = []
            video_number_list = []
            image_number_list = []
            for data, target, video_number, image_number in val_loader:
                predicted_list = []
                target_list = []
                if args.cuda:
                    data = data.cuda()
                output = model(data)
                multi_target = target.permute(1, 0)
                video_number = video_number
                image_number = image_number
                multi_output = output
                for sequence in range(args.sequence):
                    target = multi_target[sequence].cuda()
                    output = multi_output[sequence]
                    predicted_ver2 = []
                    sigmoid = nn.Sigmoid()
                    outputs_sigmoid = sigmoid(output)
                    for s in outputs_sigmoid:
                        predicted_ver2.append(float(s[1]))
                    predicted_list.append(predicted_ver2)
                    target_list.append(target.tolist())
                t.update(1)
                predicted_list = torch.Tensor(predicted_list).permute(1,0)
                predicted_list = torch.Tensor(predicted_list).reshape(args.val_batch_size*args.sequence)
                target_list = torch.Tensor(target_list).permute(1, 0)
                target_list = torch.Tensor(target_list).reshape(args.val_batch_size*args.sequence)
                video_number = video_number.reshape(args.val_batch_size*args.sequence)
                image_number = image_number.reshape(args.val_batch_size*args.sequence)
                predicted_multi_list += predicted_list.tolist()
                target_multi_list += target_list.tolist()
                video_number_list += video_number.tolist()
                image_number_list += image_number.tolist()

            predicted_multi_list = [float(i) for i in predicted_multi_list]
            target_multi_list = [int(i) for i in target_multi_list]
            eval_res = select_keyshots(predicted_multi_list, video_number_list, image_number_list, target_multi_list, args)
            fscore_k = 0
            for i in eval_res:
                fscore_k+=i[2]
            fscore_k/= len(list(args.test_dataset.split(",")))

            pd_F_measure_k.append(fscore_k)

    save_model(model, args, fscore_k, epoch)
    print("test video number:")
    print(args.test_dataset)
    print("F_measure_k:")
    print(fscore_k)




def train(model, train_loader, optimizer, criterion, epoch, args):

    global pd_lr
    global pd_loss

    train_loss = Metric('train_loss')
    model.train()
    N = len(train_loader)
    start_time = time.time()

    for batch_idx, (data, target, video_number, image_number) in enumerate(train_loader):
        lr_cur = adjust_learning_rate(args, optimizer, epoch, batch_idx, N, type=args.lr_scheduler)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        output = model(data)
        multi_target = target.permute(1, 0)
        multi_output = output
        multi_loss = 0
        for sequence in range(args.sequence):
            target = multi_target[sequence].cuda()
            output = multi_output[sequence]
            loss = criterion(output, target)
            multi_loss+=loss
            train_loss.update(loss)
        multi_loss/=args.sequence
        multi_loss.backward()
        optimizer.step()



        if (batch_idx + 1) % 100 == 0:
            memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
            used_time = time.time() - start_time
            eta = used_time / (batch_idx + 1) * (N - batch_idx)
            eta = str(datetime.timedelta(seconds=int(eta)))
            training_state = '  '.join(
                [
                    'Epoch: {}',
                    '[{} / {}]',
                    'eta: {}',
                    'lr: {:.9f}',
                    'max_mem: {:.0f}',
                    'loss: {:.3f}',
                ]
            )
            training_state = training_state.format(
                epoch + 1,
                batch_idx + 1,
                N,
                eta,
                lr_cur,
                memory,
                train_loss.avg.item(),
            )
            print(training_state)

        if batch_idx == N - 1:
            pd_lr.append(lr_cur)
            pd_loss.append(train_loss.avg.item())

def train_net(args):
    print("Dataset:")
    print(args.dataset)
    print("---Initializing---")
    train_loader, val_loader,In_target = build_dataloader(args)
    total_target = len(train_loader)*args.batch_size*args.sequence
    A = total_target/(total_target-In_target)
    B = total_target/In_target
    model = build_model(args)
    print('Parameters:', sum([np.prod(p.size()) for p in model.parameters()]))
    optimizer = build_optimizer(args, model)

    epoch = 0
    if args.resume:
        epoch = resume_model(model, optimizer, args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    cudnn.benchmark = True

    if args.cuda:
        model.cuda()


    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([A, B])).to(device)


    print("---Training Start---")


    global pd_epoch
    global pd_batch_size
    global pd_lr
    global pd_runtime
    global pd_loss
    global pd_F_measure_k
    

    while epoch < args.epochs:
        pd_epoch.append(epoch)
        pd_batch_size .append(args.batch_size)
        Stime = time.time()
        train(
            model, train_loader, optimizer, criterion, epoch, args
        )
        if (epoch + 1) % args.test_epochs == 0:

            val(model, val_loader, epoch, args)

        Etime = time.time()
        runtime = str(datetime.timedelta(seconds=int(Etime - Stime)))
        pd_runtime.append(runtime)

        ddict = {'epoch': pd_epoch,
                 'Batch_size':pd_batch_size,
                 'lr':pd_lr,
                 'runtime':pd_runtime,
                 'loss':pd_loss,
                 'F_measure_k':pd_F_measure_k
                 }

        dataframe = pd.DataFrame(ddict)
        csv_path = "./STSA/exprecord_dirs/CSV/"+args.dataset+"/Record_" + str(args.exp_name) + ".csv"
        dataframe.to_csv(csv_path, index=False, sep=',')

        epoch += 1




def run_command(command):
    p = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return '\n'.join([ '\t'+line.decode("utf-8").strip() for line in p.stdout.readlines()])

if __name__ == "__main__":
    args = parse_args()

    dep_versions = {}
    dep_versions['display'] = run_command('cat /proc/driver/nvidia/version')

    dep_versions['cuda'] = 'NA'
    cuda_home = '/usr/local/cuda/'
    if 'CUDA_HOME' in os.environ:
        cuda_home = os.environ['CUDA_HOME']

    cmd = cuda_home+'/version.txt'
    if os.path.isfile(cmd):
        dep_versions['cuda'] = run_command('cat '+cmd)

    dep_versions['cudnn'] = torch.backends.cudnn.version()
    dep_versions['platform'] = platform.platform()
    dep_versions['python'] = sys.version_info[:3]
    dep_versions['torch'] = torch.__version__
    dep_versions['numpy'] = np.__version__
    dep_versions['h5py'] = h5py.__version__
    dep_versions['json'] = json.__version__
    dep_versions['ortools'] = ortools.__version__
    dep_versions['torchvision'] = pkg_resources.get_distribution("torchvision").version

    for key, value in dep_versions.items():
        print(key,": ",value)
    print("")


    train_net(args)
