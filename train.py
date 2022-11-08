import torch
from models import get_model
import time
from datasets import get_dataloader
import numpy as np
from torch import nn
import os
import sys
from config import TrainConfig
import fire
from attacks import get_attack
import logging
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

    
def train(**kwargs):
    # load and adjust configs
    args = TrainConfig()
    args.parse(kwargs)
    
    # set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    # make a new file folder (named `file_path`) to store checkpoints
    curr = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    folder_name = f'{args.dataset_name}_{args.model_name}_bs{args.train_batchsize}_lr{args.lr}'
    if args.adv_train:
        folder_name = f'{args.dataset_name}_{args.model_name}_adv_bs{args.train_batchsize}_lr{args.lr}'
    
    file_path = os.path.join(args.out_ckpt_folder, folder_name)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        
    
    
    # define the log file that receives your log info
    log_file_name = folder_name+f"_{curr}.log"
    subfile_path = os.path.join(file_path, folder_name+f"_{curr}")
    os.makedirs(subfile_path)
    
    # define the log file that receives your log info
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.INFO)
    log_file_name = folder_name+f"_{curr}.log"
    # file stream
    fh = logging.FileHandler(os.path.join(subfile_path, log_file_name), encoding='utf8')
    logger.addHandler(fh)
    # screen stream
    sh = logging.StreamHandler()
    logger.addHandler(sh)
    
    logger.info(f'Writing to {os.path.join(subfile_path, log_file_name)}')
    
    # print the config file
    logger.info('Config file:')
    logger.info(args)
    logger.info('')
    
    # get device
    gpu_list = [int(i) for i in args.gpu.strip().split(",")]
    device = torch.device(f"cuda:{gpu_list[0]}" if torch.cuda.is_available() else "cpu")

    # get dataloaders
    train_dataloader = get_dataloader(mode='train', dataset_name=args.dataset_name, dataset_path=args.dataset_path[args.dataset_name], batchsize=args.train_batchsize, num_workers=args.num_workers)
    val_dataloader = get_dataloader(mode='val', dataset_name=args.dataset_name, dataset_path=args.dataset_path[args.dataset_name], batchsize=args.train_batchsize, num_workers=args.num_workers)

    # create a model
    model = get_model(model_name=args.model_name, dataset_name=args.dataset_name, device=device, ckpt_path=args.ckpt_path)

    # define loss function
    loss_fn = nn.CrossEntropyLoss()

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_period, gamma=args.lr_decay)
    
    # continue to train from the existed checkpoints
    start_epoch = 1
    if args.ckpt_path:
        import re
        searchObj = re.search( r'epoch_(.*).pth', args.ckpt_path, re.M|re.I)
        start_epoch = int(searchObj.group(1))+1
        logger.info(f'Retrieved checkpoints from {args.ckpt_path}')
    
    train_time = 0
    val_train = 0
    
    writer = SummaryWriter(log_dir = '/root/tf-logs/')
    
    # train!
    for t in range(start_epoch, args.epochs+1):
        logger.info(f"Epoch {t}\n-------------------------------")
        
        tic = time.time()
        # training process
        # train(args.adv_train, train_dataloader, model, loss_fn, optimizer, device)
        size = len(train_dataloader.dataset)
        model.train()
        for batch, (clean_images, labels, target_labels) in tqdm(enumerate(train_dataloader)):
            clean_images, labels = clean_images.to(device), labels.to(device)
            target_labels = target_labels.to(device)
            
            if args.adv_train:
                adversary = get_attack(attack_name=args.attack_name, model=model, eps=args.eps[0], nb_iter=args.nb_iter[0])
                adv_images = adversary.perturb(clean_images, labels)
                # Compute prediction error
                adv_pred = model(adv_images)
                loss = loss_fn(adv_pred, labels)
                
            else:
                # Compute prediction error
                pred = model(clean_images)
                loss = loss_fn(pred, labels)
            
            
            # add loss to the tensorboard
            writer.add_scalar(tag = "loss/train", scalar_value = loss, global_step = (t-1) * len(train_dataloader) + batch)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if batch % 10 == 0:
            #     loss, current = loss.item(), batch * len(clean_images)
            #     logger.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        toc = time.time()
        train_time += toc - tic
        
        # validating process
        # val(args.adv_train, val_dataloader, model, loss_fn, device)
        size = len(val_dataloader.dataset)
        num_batches = len(val_dataloader)
        model.eval()
        test_loss, correct = 0, 0
        adv_test_loss, adv_correct = 0, 0
        if args.adv_train:
            for batch, (clean_images, labels, target_labels) in tqdm(enumerate(val_dataloader)):
                clean_images, labels = clean_images.to(device), labels.to(device)
                if args.adv_train:
                    adversary = get_attack(attack_name=args.attack_name, model=model, eps=args.eps[0], nb_iter=args.nb_iter[0])
                    adv_images = adversary.perturb(clean_images, labels)
                    # Compute prediction error
                    adv_pred = model(adv_images)
                    adv_test_loss += loss_fn(adv_pred, labels).item()
                    adv_correct += (adv_pred.argmax(1) == labels).type(torch.float).sum().item()
                # Compute prediction error
                pred = model(clean_images)
                test_loss += loss_fn(pred, labels).item()
                correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
                
        else:
            with torch.no_grad():
                for batch, (clean_images, labels, target_labels) in enumerate(val_dataloader):
                    clean_images, labels = clean_images.to(device), labels.to(device)
                    # Compute prediction error
                    pred = model(clean_images)
                    test_loss += loss_fn(pred, labels).item()
                    correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
                    
        test_loss /= num_batches
        correct /= size
        
        # add loss and accuracy to the tensorboard
        writer.add_scalar(tag = "loss/val", scalar_value = test_loss, global_step = t * len(train_dataloader))
        writer.add_scalar(tag = "acc/val", scalar_value = correct, global_step = t * len(train_dataloader))
        
        val_time = time.time() - toc
        
        logger.info(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        if args.adv_train:
            adv_test_loss /= num_batches
            adv_correct /= size
            logger.info(f"Adv test Error: \n Accuracy: {(100*adv_correct):>0.1f}%, Avg loss: {adv_test_loss:>8f} \n")
        
        
        # save the checkpoints
        if t % 10 == 0:
            ckpt_name = f'epoch_{t}.pth'
            torch.save(model.state_dict(), os.path.join(subfile_path, ckpt_name))
            logger.info(f"Saved PyTorch Model State to {os.path.join(subfile_path, ckpt_name)}")
            
    writer.close()
    logger.info("Done!")
    
    logger.info('Average train speed: {:.2f} s per epoch \n'.format(train_time/(args.epochs+1-start_epoch)))   
    logger.info('Average val speed: {:.2f} s per epoch \n'.format(val_time/(args.epochs+1-start_epoch)))   


if __name__ == '__main__':
    fire.Fire()