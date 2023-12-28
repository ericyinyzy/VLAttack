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
import importlib
import json
import tensorflow as tf
import tensorflow_hub as hub
from models.blip_pretrain import blip_pretrain
import sys
sys.path.append('../cleverhans')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.backends.cudnn as cudnn
from models.blip_vqa import blip_vqa
import utils
from data import create_dataset, create_sampler, create_loader
from data.vqa_dataset import vqa_collate_fn
from transformers import BertTokenizer


def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer


def main(args, config, config_pretrain):
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
    train_loader, test_loader = create_loader(datasets, samplers,
                                              batch_size=[config['batch_size_train'], config['batch_size_test']],
                                              num_workers=[16, 16], is_trains=[True, False],
                                              collate_fns=[vqa_collate_fn, None])
    #### Model ####
    print("Creating model")
    model = blip_vqa(pretrained=config['pretrained'], image_size=config['image_size'],
                     vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])

    model = model.to(device)
    pretrain_model = blip_pretrain(pretrained=config_pretrain['pretrained'], image_size=config_pretrain['image_size'],
                                   vit=config_pretrain['vit'], vit_grad_ckpt=config['vit_grad_ckpt'],
                                   vit_ckpt_layer=config_pretrain['vit_ckpt_layer'],
                                   queue_size=config_pretrain['queue_size'])
    pretrain_model = pretrain_model.to(device)
    tokenizer = init_tokenizer()
    with tf.device('cpu'):
        USE_model = hub.load(args.USE_model_path)
    f = open(args.correct_idx_store, 'r')
    a = list(f)
    f.close()
    correct_list = [int(l.strip('\n')) for l in a][:5000]
    with open(args.correct_pred_store, 'r') as f:
        answer_list = json.load(f)

    pgd_attack = Adv_attack(model, pretrain_model, tokenizer, device, correct_list,answer_list,USE_model)
    pgd_attack.evaluate(test_loader, tokenizer)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_pre', default='./configs/pretrain.yaml')
    parser.add_argument('--config', default='./configs/vqa.yaml')
    parser.add_argument('--output_dir', default='output/VQA')
    parser.add_argument('--correct_idx_store', default='right_vqa_list.txt')
    parser.add_argument('--USE_model_path', default='https://tfhub.dev/google/universal-sentence-encoder-large/5')
    parser.add_argument('--correct_pred_store', default='right_vqa_ans_table.txt')
    parser.add_argument('--method', default='',type=str)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    args = parser.parse_args()
    if args.method=='BERTAttack':
        module=importlib.import_module('attack.VQA.adv_attack_BERTattack')
        Adv_attack=getattr(module,'Adv_attack')
    elif args.method=='BSA':
        module = importlib.import_module('attack.VQA.adv_attack_blip')
        Adv_attack =getattr(module,'Adv_attack')
    elif args.method=='Co-Attack':
        module = importlib.import_module('attack.VQA.adv_attack_coattack')
        Adv_attack =getattr(module,'Adv_attack')
    elif args.method == 'VLAttack':
        module = importlib.import_module('attack.VQA.adv_attack_blip_vla')
        Adv_attack =getattr(module,'Adv_attack')
    else:
        print('attack method is not supported!')
        raise ValueError


    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    config_pretrain = yaml.load(open(args.config_pre, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    main(args, config, config_pretrain)