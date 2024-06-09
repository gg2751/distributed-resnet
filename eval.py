import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import time
import multiprocessing
from argparse import ArgumentParser

from model import ResNet18
from model import ResNet18NoBN

def main():
    parser = ArgumentParser(description="Evaluation of Training Settings using ResNet18")
    parser.add_argument('--use_cuda', type=bool, default=True, help='Use CUDA, if required, default=True')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to Dataset, default: ./data')
    parser.add_argument('--optimizer', type=str, default='sgd', help='Select Optimizer: SGD, Nesterov, Adagrad, Adadelta, Adam, default=SGD')
    parser.add_argument("--batch_size", type=int, default=128, help='Select Batch Size, recommended: 32, 64, 128, default=128')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of Workers, default=2') 
    parser.add_argument("--num_gpus", type=int, default=1, help='Number of GPUs if required, default=1')
    parser.add_argument("--get_bandwidth", type=bool, default=False, help='Calculate set up bandwidth, default=False')

    args = parser.parse_args()

    # Set Device
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    
    print('==> Preparing data')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), 
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                            (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    testset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    print('==> Building model')
    net = ResNet18()
    # net = ResNet18NoBN()     # Uncomment to use ResNet18 without Batch Normalization
    net = net.to(device)
    # Handling multi-GPU training
    if device == 'cuda':
        total_batch_size = args.batch_size*args.num_gpus
        if args.num_gpus == 1:
            net = torch.nn.DataParallel(net)
        if args.num_gpus == 2:
            net = torch.nn.DataParallel(net, device_ids = [0,1])
        elif args.num_gpus == 3:
            net = torch.nn.DataParallel(net, device_ids = [0,1,2])
        elif args.num_gpus == 4:
            net = torch.nn.DataParallel(net, device_ids = [0,1,2,3])
    cudnn.benchmark = True

    # Set Optimizer
    if args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    elif args.optimizer.lower() == "nesterov":
        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    elif args.optimizer.lower() == "adagrad":
        optimizer = optim.Adagrad(net.parameters(), lr=0.1, weight_decay=5e-4)
    elif args.optimizer.lower() == "adadelta":
        optimizer = optim.Adadelta(net.parameters(), lr=0.1, weight_decay=5e-4)
    elif args.optimizer.lower() == "adam":
        optimizer = optim.Adam(net.parameters(), lr=0.1, weight_decay=5e-4)
    else:
        raise ValueError("Invalid Optimizer\nChoose from:\nSGD\nNesterov\nAdagrad\nAdadelta\nAdam")

    criterion = nn.CrossEntropyLoss()

    def train(epoch):
        loadTime, trainTime, runningTime = 0, 0, 0
        
        net.train()
        train_loss = 0
        correct, total = 0, 0

        print(f'\nEpoch: {epoch+1}')
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            runningTime_start = time.perf_counter() 

            # Measuring data loading time
            loadTime_start = time.perf_counter() 
            inputs, targets = inputs.to(device), targets.to(device)
            torch.cuda.synchronize()
            loadTime_end = time.perf_counter() 

            # Measuring training time
            trainTime_start = time.perf_counter() 
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            torch.cuda.synchronize()
            trainTime_end  = time.perf_counter() 

            runningTime_end = time.perf_counter() 

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            loadTime += loadTime_end - loadTime_start
            trainTime += trainTime_end - trainTime_start
            runningTime += runningTime_end - runningTime_start

        average_loss = train_loss / (batch_idx + 1)
        accuracy = 100.0 * correct / total

        print(f'{batch_idx}/{len(trainloader)} Loss: {average_loss:.3f} | Acc: {accuracy:.3f}% ({correct}/{total})')
        print(f'Data-loading time: {loadTime:.4f}sec, Training time: {trainTime:.4f}sec, Total running time: {runningTime:.4f}sec')
        return loadTime, trainTime, runningTime

    total_LoadTime = 0
    total_TrainTime = 0
    total_RunningTime = 0
    for epoch in range(5):
        loadTime, trainTime, runningTime = train(epoch)
        total_LoadTime += loadTime
        total_TrainTime += trainTime
        total_RunningTime += runningTime

    average_time = total_LoadTime/5, total_TrainTime/5, total_RunningTime/5
    print(f'Averages over 5 Epochs\n Load time: {average_time[0]:.4f}sec, Train time: {average_time[1]:.4f}sec, Total running time: {average_time[2]:.4f}sec')


    # Bandwidth calculation
    if device == "cuda" and args.get_bandwidth == True:
        if args.num_gpus > 1:
            
            torch.cuda.empty_cache()
            for epoch in range(5):      
                single_gpuTime = train(epoch)
            
            communicationTime = runningTime - single_gpuTime[2] / args.num_gpus
            computationTime   = runningTime - communicationTime
            
            bandwidth = (4162.6252 * (args.num_gpus - 1)) / (args.batch_size * (args.num_gpus) * communicationTime)
            print(f"Communication Time: {computationTime:.4f} sec \nComputation Time: {computationTime:.4f} sec \nBandwidth Utilization: {bandwidth:.4f} GB/sec")
        else:
            raise ValueError("Set num_gpus > 1 to calculate bandwidth")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
