'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
from pathlib import Path
import json
import torch
import torch.backends.cudnn as cudnn
from models.blip_nlvr import blip_nlvr
import utils
from data import create_dataset, create_sampler, create_loader

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

@torch.no_grad()
def evaluation(model, data_loader, device, config,correct_idx_store,correct_pred_store):
    model.eval()
    blip_nlvr_ans_table = {}
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    print_freq = 50
    index=0
    right_list_blip_nlvr = []
    for n,batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        index += 1
        images = torch.cat([batch['image0'], batch['image1']], dim=0)
        images, targets = images.to(device), batch['label'].to(device)
        prediction = model(images, batch['sentence'], targets=targets, train=False)
        _, pred_class = prediction.max(1)
        if pred_class == targets:
            right_list_blip_nlvr.append(int(index))
            blip_nlvr_ans_table[str(int(index))] = int(pred_class.detach().cpu().numpy()[0])
        if len(right_list_blip_nlvr) > 6000:
            break
    f = open(correct_idx_store, "w")
    for i in right_list_blip_nlvr:
        f.write(str(i) + '\n')
    f.close()
    with open(correct_pred_store, 'w') as file:
        file.write(json.dumps(blip_nlvr_ans_table))
    print(f'correct answer list and all correct predictions are stored in {correct_idx_store} and {correct_pred_store}')
        
def main(args, config,config_pretrain):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    #### Dataset #### 
    print("Creating dataset")
    datasets = create_dataset('nlvr', config) 
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True,False,False], num_tasks, global_rank)
    else:
        samplers = [None, None, None]
    
    batch_size=[config['batch_size_train'],config['batch_size_test'],config['batch_size_test']]
    train_loader, val_loader, test_loader = create_loader(datasets,samplers,batch_size=batch_size,
                                                          num_workers=[16,16,16],is_trains=[True,False,False],
                                                          collate_fns=[None,None,None])

    #### Model #### 
    print("Creating model")
    model = blip_nlvr(pretrained=config['pretrained'], image_size=config['image_size'], 
                         vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])
    model = model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
    correct_idx_store=args.correct_idx_store
    correct_pred_store=args.correct_pred_store
    evaluation(model_without_ddp, test_loader, device, config,correct_idx_store,correct_pred_store)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_pre', default='./configs/pretrain.yaml')
    parser.add_argument('--config', default='./configs/nlvr.yaml')
    parser.add_argument('--correct_idx_store', default='right_nlvr_list.txt')
    parser.add_argument('--correct_pred_store', default='right_nlvr_ans_table.txt')
    parser.add_argument('--output_dir', default='output/NLVR')
    parser.add_argument('--evaluate', action='store_true')      
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config_pretrain = yaml.load(open(args.config_pre, 'r'), Loader=yaml.Loader)
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config,config_pretrain)