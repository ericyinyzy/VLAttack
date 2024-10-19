# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import numpy as np
import sys
from typing import Dict, Iterable, Optional
import copy
import torch
import torch.nn
import torch.optim
import os
import matplotlib.pyplot as plt
import util.dist as dist
from datasets.coco_eval import CocoEvaluator
from datasets.flickr_eval import FlickrEvaluator, FlickrCaptionEvaluator
from datasets.refexp import RefExpEvaluator
from util.metrics import MetricLogger, SmoothedValue
from util.misc import targets_to
from util.optim import adjust_learning_rate, update_ema
import json
sys.path.append('../cleverhans')
import cleverhans.torch.attacks.projected_gradient_descent as pgd
def train_one_epoch(
    model: torch.nn.Module,
    criterion: Optional[torch.nn.Module],
    weight_dict: Dict[str, float],
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args,
    max_norm: float = 0,
    model_ema: Optional[torch.nn.Module] = None,
):
    model.train()
    if criterion is not None:
        criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("lr_backbone", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("lr_text_encoder", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 1000

    num_training_steps = int(len(data_loader) * args.epochs)
    for i, batch_dict in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        curr_step = epoch * len(data_loader) + i
        samples = batch_dict["samples"].to(device)
        positive_map = batch_dict["positive_map"].to(device) if "positive_map" in batch_dict else None
        targets = batch_dict["targets"]
        answers = {k: v.to(device) for k, v in batch_dict["answers"].items()} if "answers" in batch_dict else None
        captions = [t["caption"] for t in targets]

        targets = targets_to(targets, device)

        memory_cache = model(samples, captions, targets, encode_and_save=True)
        outputs = model(samples, captions, targets, encode_and_save=False, memory_cache=memory_cache)

        loss_dict = {}
        if criterion is not None:
            loss_dict.update(criterion(outputs, targets, positive_map))

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = dist.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        adjust_learning_rate(
            optimizer,
            epoch,
            curr_step,
            num_training_steps=num_training_steps,
            args=args,
        )
        if model_ema is not None:
            update_ema(model, model_ema, args.ema_decay)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(lr_backbone=optimizer.param_groups[1]["lr"])
        metric_logger.update(lr_text_encoder=optimizer.param_groups[2]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
class Adv_attack:
    def __init__(self, model,device,white_box=None,black_box=None):
        self.white_model=copy.deepcopy(model)
        checkpoint = torch.load(white_box, map_location="cpu")
        state_dict = checkpoint["model_ema"]
        self.white_model.load_state_dict(state_dict, strict=False)
        self.black_model=copy.deepcopy(model)
        checkpoint = torch.load(black_box, map_location="cpu")
        state_dict = checkpoint["model_ema"]
        self.black_model.load_state_dict(state_dict, strict=False)
        self.device=device
        self.batch=None
        self.captions=None
        self.l2 = torch.nn.MSELoss()
        self.std=[0.229, 0.224, 0.225]
        self.length=[]
        self.idx1=0
        # if not os.path.exists(self.sourth_path):
        #     os.makedirs(self.sourth_path)
    def pgd_attack(self,x):
        if self.batch is None or self.captions is None:
            raise ValueError
        samples = self.batch["samples"].to(self.device)
        targets = self.batch["targets"]
        # answers = {k: v.to(device) for k, v in batch_dict["answers"].items()} if "answers" in batch_dict else Non
        targets = targets_to(targets, self.device)
        samples.tensors = x
        memory_cache = self.white_model(samples, self.captions, targets, encode_and_save=True)
        text_masks = torch.where(memory_cache['mask'][0] == False)
        ori_enc_feats = memory_cache['enc_feats'][:,text_masks[0], 0, :]
        ori_res_feats = memory_cache['ori_res_feats']
        return [ori_res_feats,ori_enc_feats]
    def norm_img(self,img):
        img = (img.permute(0, 2, 3, 1).cpu() * torch.tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(0).unsqueeze(
            0)) + torch.tensor([0.485, 0.456, 0.406])
        if torch.min(img) < 0:
            img = (img - torch.min(img))
            img = img / torch.max(img)
        else:
            img = (img + torch.min(img))
            img = img / torch.max(img)
        return img
    @torch.no_grad()
    def evaluate(
        self,
        criterion: Optional[torch.nn.Module],
        postprocessors: Dict[str, torch.nn.Module],
        weight_dict: Dict[str, float],
        data_loader,
        evaluator_list,
        args,
    ):
        self.white_model.eval()

        self.black_model.eval()
        if criterion is not None:
            criterion.eval()


        metric_logger = MetricLogger(delimiter="  ")
        header = "Test:"

        f = open('right_refcoco_5k.txt', 'r')
        a = list(f)
        f.close()
        correct_list = [int(l.strip('\n')) for l in a][:5000]
        idx=0
        success_count=0
        multi_num=0
        iter_step = 0
        iter_dict={}

        import json
        with open('all_adv_bert_attack_bank_on_refcoco_ori_90.txt', 'r') as f:
            text_bank = json.load(f)
        
        for batch_dict in metric_logger.log_every(data_loader, 100, header):
        # exit()

            samples = batch_dict["samples"].to(self.device)
            positive_map = batch_dict["positive_map"].to(self.device) if "positive_map" in batch_dict else None

            targets = batch_dict["targets"]
            # answers = {k: v.to(self.device) for k, v in batch_dict["answers"].items()} if "answers" in batch_dict else None
            captions = [t["caption"] for t in targets]
            if targets[0]["image_id"].item() not in correct_list:
                continue
            idx += 1
            if idx %10==0:
                print('refcoco_acc:', success_count/idx,success_count,idx)#,multi_num,iter_step)
            # if idx>100:
            #     break
            ori_img=copy.deepcopy(samples.tensors)
            adv_img = copy.deepcopy(samples.tensors)
            clip_max = float(torch.max(adv_img).detach().cpu().numpy())
            clip_min = float(torch.min(adv_img).detach().cpu().numpy())
            perturb_budet = (4 / 256)/0.229 # normalized std = [0.229, 0.224, 0.225]
            self.batch = copy.deepcopy(batch_dict)
            self.captions=copy.deepcopy(captions)
            targets = targets_to(targets, self.device)
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            memory_cache= self.white_model(samples, captions, targets, encode_and_save=True)
            text_masks = torch.where(memory_cache['mask'][0] == False)
            ori_enc_feats = memory_cache['enc_feats'][:,text_masks[0], 0, :]
            ori_res_feats=memory_cache['ori_res_feats']
            torch.set_grad_enabled(True)
            adv_x, _ = pgd.projected_gradient_descent(self.pgd_attack, adv_img, perturb_budet, 0.01, 20, np.inf, clip_min=clip_min,
                                                         clip_max=clip_max, y=[ori_res_feats, ori_enc_feats], time=0,
                                                         ori_x=ori_img)
            torch.set_grad_enabled(False)
            samples.tensors=adv_x
            adv_img=adv_x
            memory_cache = self.black_model(samples, captions, targets, encode_and_save=True)
            outputs = self.black_model(samples, captions, targets, encode_and_save=False, memory_cache=memory_cache)
            results = postprocessors["bbox"](outputs, orig_target_sizes)
            if results[0]['boxes'].shape[0] == 1:
                for result in results:
                    result['scores'] = result['scores'].unsqueeze(1)[0]
                    result['labels'] = result['labels'].unsqueeze(1)[0]
            res = {target["image_id"].item(): output for target, output in zip(targets, results)}
            evaluator_list[-1].update(res)
            refexp_res, success,box = evaluator_list[-1].summarize()

            if success==1:
                success_count+=1
                continue
            else:
                samples.tensors = copy.deepcopy(ori_img)
                if int(text_bank[str(targets[0]["image_id"].item())][-1]) == 1:
                    print('text_success')
                    adv_text = text_bank[str(targets[0]["image_id"].item())][-2]
                    memory_cache = self.black_model(samples, [adv_text], targets, encode_and_save=True)
                    outputs = self.black_model(samples, [adv_text], targets, encode_and_save=False,
                                               memory_cache=memory_cache)
                    results = postprocessors["bbox"](outputs, orig_target_sizes)
                    if results[0]['boxes'].shape[0] == 1:
                        for result in results:
                            result['scores'] = result['scores'].unsqueeze(1)[0]
                            result['labels'] = result['labels'].unsqueeze(1)[0]
                    res = {target["image_id"].item(): output for target, output in zip(targets, results)}
                    evaluator_list[-1].update(res)
                    refexp_res, success,box = evaluator_list[-1].summarize()
                    if success == 0:
                        print('wrong_prediction')
                        raise ValueError
                    success_count+=1
                    continue
                else:
                    text_bank_sort = text_bank[str(targets[0]["image_id"].item())][:-1]
                    if len(text_bank_sort) > 20:
                        text_bank_sort = text_bank_sort[:20]
                    if len(text_bank_sort) == 0:
                        text_bank_sort = text_bank_sort + [captions[0]]
                    iters = int((40 - 20) / len(text_bank_sort))
                    iters_list = [iters for i in range(len(text_bank_sort))]
                    if 20 - sum(iters_list) < 0:
                        raise ValueError
                    iters_list[-1] += 20 - sum(iters_list)
                    if 20 - sum(iters_list) < 0:
                        print('more than 20')
                        raise ValueError
                    adv_img_txt = adv_img
                    loss_list = []
                    sort_bank = copy.deepcopy(text_bank_sort)
                    idxx = 0
                    ii = 0
                    max_len = len(text_bank_sort) - 1
                    count_iter=0

                    while True:
                        if idxx > max_len:
                            break
                        else:
                            self.captions = copy.deepcopy([sort_bank[idxx]])
                            iters = iters_list[idxx]
                            idxx += 1
                            count_iter+=iters
                        samples.tensors = adv_img_txt
                        memory_cache = self.black_model(samples, self.captions, targets, encode_and_save=True)
                        outputs = self.black_model(samples, self.captions, targets, encode_and_save=False,
                                                   memory_cache=memory_cache)
                        results = postprocessors["bbox"](outputs, orig_target_sizes)
                        if results[0]['boxes'].shape[0] == 1:
                            for result in results:
                                result['scores'] = result['scores'].unsqueeze(1)[0]
                                result['labels'] = result['labels'].unsqueeze(1)[0]
                        res = {target["image_id"].item(): output for target, output in zip(targets, results)}
                        evaluator_list[-1].update(res)
                        refexp_res, success, box = evaluator_list[-1].summarize()
                        if success == 1:
                            success_count += 1
                            break
                        torch.set_grad_enabled(True)
                        adv_x, _ = pgd.projected_gradient_descent(self.pgd_attack, adv_img_txt, perturb_budet, 0.01, iters, np.inf, clip_min=clip_min,
                                                         clip_max=clip_max, y=[ori_res_feats, ori_enc_feats], time=1,
                                                         ori_x=ori_img)
                        torch.set_grad_enabled(False)

                        samples.tensors = adv_x
                        memory_cache = self.black_model(samples, self.captions, targets, encode_and_save=True)
                        outputs = self.black_model(samples, self.captions, targets, encode_and_save=False,
                                                   memory_cache=memory_cache)
                        results = postprocessors["bbox"](outputs, orig_target_sizes)
                        if results[0]['boxes'].shape[0] == 1:
                            for result in results:
                                result['scores'] = result['scores'].unsqueeze(1)[0]
                                result['labels'] = result['labels'].unsqueeze(1)[0]
                        res = {target["image_id"].item(): output for target, output in zip(targets, results)}
                        evaluator_list[-1].update(res)
                        refexp_res, success,box = evaluator_list[-1].summarize()
                        if success== 1:
                            success_count += 1
                            break
                        adv_img_txt = adv_x
                        ii += 1
        print('final_refcoco_asr:', success_count/idx,success_count,idx)
    
        exit()
    # gather the stats from all processes
