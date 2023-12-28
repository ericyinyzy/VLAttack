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
# from models.med import BertTokenizer
import json
from pathlib import Path

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.backends.cudnn as cudnn
#45。52
#26。56

from models.blip_vqa import blip_vqa
import utils
from data import create_dataset, create_sampler, create_loader
from data.vqa_dataset import vqa_collate_fn


@torch.no_grad()
def evaluation(model, data_loader, device, config,correct_idx_store,correct_pred_store) :
    # test
    model.eval()
    blip_ans_table={}
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 50
    result = []
    right_list_blip = []
    if config['inference']=='rank':   
        answer_list = data_loader.dataset.answer_list
        answer_candidates = model.tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)    
        answer_candidates.input_ids[:,0] = model.tokenizer.bos_token_id
    correct=0
    for n, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        correct += 1
        image = batch['image'].to(device,non_blocking=True)
        result = []
        if config['inference']=='generate':
            answers = model(batch['image'], batch['question'], train=False, inference='generate')
            for answer, ques_id in zip(answers, batch['question_id']):
                ques_id = int(ques_id.item())       
                result.append({"question_id":ques_id, "answer":answer})
        elif config['inference']=='rank':    
            answer_ids = model(image, batch['question'], answer_candidates, train=False, inference='rank', k_test=config['k_test'])
            for ques_id, answer_id in zip(batch['question_id'], answer_ids):
                result.append({"question_id":int(ques_id.item()), "answer":answer_list[answer_id]})
        answer = [i[0] for i in batch['answer']]
        weights = [float(i[0].cpu().numpy()) for i in batch['weight']]
        pred_ans = result[0]['answer']
        if pred_ans in answer:
            if weights[answer.index(pred_ans)] ==max(weights):
                right_list_blip.append(int(batch['question_id'][0]))
                blip_ans_table[str(int(batch['question_id'][0]))] = pred_ans
        if len(right_list_blip)>6000:
            break
    f = open(correct_idx_store, "w")
    for i in right_list_blip:
        f.write(str(i) + '\n')
    f.close()
    with open(correct_pred_store, 'w') as file:
        file.write(json.dumps(blip_ans_table))
    print(len(right_list_blip))
    print(f'correct answer list is stored in ---- {correct_idx_store}.')
    print(f'All correct predictions are stored in ---- {correct_pred_store}')
    return result


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
    print("Creating vqa datasets")
    datasets = create_dataset('vqa', config)   
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)         
    else:
        samplers = [None, None]
    
    train_loader, test_loader = create_loader(datasets,samplers,
                                              batch_size=[config['batch_size_train'],config['batch_size_test']],
                                              num_workers=[16,16],is_trains=[True, False],
                                              collate_fns=[vqa_collate_fn,None])

    #### Model #### 
    print("Creating model")
    model = blip_vqa(pretrained=config['pretrained'], image_size=config['image_size'], 
                       vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])
    model = model.to(device)
    model_without_ddp = model
    correct_idx_store = args.correct_idx_store
    correct_pred_store = args.correct_pred_store
    evaluation(model_without_ddp, test_loader, device, config,correct_idx_store,correct_pred_store)
    
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_pre', default='./configs/pretrain.yaml')
    parser.add_argument('--config', default='./configs/vqa.yaml')
    parser.add_argument('--correct_idx_store', default='right_vqa_list.txt')
    parser.add_argument('--correct_pred_store', default='right_vqa_ans_table.txt')
    parser.add_argument('--output_dir', default='output/VQA')
    parser.add_argument('--evaluate', action='store_true')      
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    config_pretrain = yaml.load(open(args.config_pre, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config,config_pretrain)