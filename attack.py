import torch
from models import get_model
import time
from datasets import get_dataloader
from defenses import get_defense
import numpy as np
from attacks import get_attack
from torchvision.utils import save_image
import os
import sys
from torch import nn
from config import AttackConfig
import pandas as pd
import fire

def attack(**kwargs):
    # load and adjust configs
    args = AttackConfig()
    args.parse(kwargs)
    
    # set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    # make a new file folder (named `file_path`) to store the output
    if args.attack_name == '':
        attack_savename = 'clean'
    else:
        attack_savename = args.attack_name
    
    if not args.defenses:
        defense_savename = 'no-defense'
    else:
        defense_savename = '+'.join(args.defenses)
    
    folder_name = f'{attack_savename}_{defense_savename}_{args.dataset_name}_{args.sub_model_name}_{args.defense_model_name}'
    file_path = os.path.join(args.out_attack_folder, folder_name)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    # define the log file that receives your log info
    curr = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    subfile_path = os.path.join(file_path, folder_name+f"_{curr}")
    os.makedirs(subfile_path)
    log_file_name = folder_name+f"_{curr}.log"
    log_file = open(os.path.join(subfile_path, log_file_name), "w")
    print(f'Writing to {os.path.join(subfile_path, log_file_name)}')
    # redirect print output to log file
    sys.stdout = log_file
    
    # print the config file
    print('Config file:')
    print(args)
    print('')
    
    # get device
    gpu_list = [int(i) for i in args.gpu.strip().split(",")]
    device = torch.device(f"cuda:{gpu_list[0]}" if torch.cuda.is_available() else "cpu")

    # get the dataloader
    adv_dataloader = get_dataloader(mode='attack', dataset_name=args.dataset_name, dataset_path=args.dataset_path[args.dataset_name], batchsize=args.attack_batchsize, num_workers=args.num_workers)

    # create a model and load weights
    sub_model = get_model(model_name=args.sub_model_name, dataset_name=args.dataset_name, device=device, ckpt_path=args.sub_ckpt_path)
    sub_model.eval()
    
    defense_model = get_model(model_name=args.defense_model_name, dataset_name=args.dataset_name, device=device, ckpt_path=args.defense_ckpt_path)
    defense_model.eval()


    # get the attack method
    adversaries = []
    
    if args.attack_name != '':
        if len(args.nb_iter) > 1:
            for nb_iter in args.nb_iter:
                adversaries.append(get_attack(attack_name=args.attack_name, model=sub_model, eps=args.eps[0], nb_iter=nb_iter))
        else:
            for eps in args.eps:
                adversaries.append(get_attack(attack_name=args.attack_name, model=sub_model, eps=eps, nb_iter=args.nb_iter[0]))
                
    success_num = [0] * len(adversaries)
    test_num = 0
    
    # get the defense ( = preprocessing) method
    defense = get_defense(args.defenses)
    
    # enumerate(xxx, 1) means counting from 1
    total_num = 0
    
    tic = time.time()
    for batch, (clean_images, labels, target_labels) in enumerate(adv_dataloader):
        bs = clean_images.shape[0]
        total_num += bs
        
        # get the clean images
        clean_images, labels = clean_images.to(device), labels.to(device)
        target_labels = target_labels.to(device)
        clean_dev_images = defense(clean_images)
        
        # inference the clean images
        clean_out = defense_model(clean_dev_images)
        clean_pred = torch.argmax(clean_out, dim=1)
        test_num += (clean_pred == labels).sum()
        
        if args.attack_name != '':
            # get the adversarial images
            for j, adversary in enumerate(adversaries):
                adv_images = adversary.perturb(clean_images, labels)
                adv_dev_images = defense(adv_images)

                # save images to show the attack effects
                if batch == 0:
                    if args.attack_name == 'fgsm':
                        filename = folder_name+"_eps{:.2f}.png".format(adversary.eps)
                    else:
                        filename = folder_name+"_iter{:0>2d}_eps{:.2f}.png".format(adversary.nb_iter, adversary.eps)
                    load_path = os.path.join(subfile_path, filename)
                    save_image(torch.cat([clean_images, clean_dev_images, adv_images, adv_dev_images], 0), load_path, nrow=bs, padding=2, normalize=True, range=(0, 1), scale_each=False, pad_value=0)

                # inference the adversarial images
                adv_out = defense_model(adv_dev_images)
                adv_pred = torch.argmax(adv_out, dim=1)

                if args.target:
                    success_num[j] += (adv_pred == target_labels).sum()
                else:
                    success_num[j] += (adv_pred != labels).sum()
            break
    toc = time.time()
    print('The total attack & defense process costs {:.2f} s'.format(toc-tic))
    print('Average speed: {:.2f} s per 100 iter \n'.format((toc-tic) / ( total_num*100 / args.attack_batchsize)))   
    final_test_acc = test_num.item() / total_num
    print("Accuracy on the clean dataset of %s: %.2f %%" % (args.dataset_name, final_test_acc * 100))
    
    
    
    # save the results to a csv file
    output_name = folder_name+f"_{curr}.csv"
    output_path = os.path.join(subfile_path, output_name)
    
    success_rates = []
    if args.attack_name != '':
        print("Success Rate of %s on the dataset of %s :" % (args.attack_name, args.dataset_name))
        if len(args.nb_iter) > 1:
            for i, iter_ in enumerate(args.nb_iter):
                success_rate = success_num[i].item() / total_num
                success_rates.append(success_rate)
                print(f'iter: {iter_}, asr: {success_rate}')
            df = pd.DataFrame({'epsilon':args.eps[0], 'iter':args.nb_iter, 'success_rates': success_rates})
        else:
            for i, eps in enumerate(args.eps):
                success_rate = success_num[i].item() / total_num
                success_rates.append(success_rate)
                print(f'eps: {eps}, asr: {success_rate}')
            df = pd.DataFrame({'epsilon':args.eps, 'iter':args.nb_iter[0], 'success_rates': success_rates})
        success_rates.append(success_rate)
    print('\n Saved to :',output_path)
    df.to_csv(output_path, index=False, sep=',')


if __name__ == '__main__':
    fire.Fire()