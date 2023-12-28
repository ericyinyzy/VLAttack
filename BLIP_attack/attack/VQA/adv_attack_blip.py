
import numpy as np
import sys

from transformers import BertForMaskedLM, BertTokenizer
from transformers.models.bert.modeling_bert import BertConfig#, BertEmbeddings
config_atk = BertConfig.from_pretrained('bert-base-uncased')
# from models.xbert import BertConfig,BertEmbeddings
from typing import Dict, Iterable, Optional
import copy
import torch
import torch.nn
import utils
import torch.optim
class Feature(object):
    def __init__(self, seq_a):
        # self.label = label
        self.seq = seq_a
        self.final_adverse = seq_a
        self.query = 0
        self.change = 0
        self.success = 0
        self.sim = 0.0
        self.changes = []
import cleverhans.torch.attacks.BLIP.projected_gradient_descent as pgd
class Adv_attack:
    def __init__(self, vqa_model,pretrain_model,tokenizer,device,correct_idx_list,correct_pred_list,USE_model):
        self.attack_dict = {}
        self.acc_list=[]
        self.tokenizer = tokenizer
        self.tokenizer_mlm = BertTokenizer.from_pretrained("bert-base-uncased",
                                                           do_lower_case="uncased" in "bert-base-uncased")
        self.total_stg_step = 40
        self.correct_list = correct_idx_list
        self.blip_ans_table=correct_pred_list
        self.white_model=pretrain_model
        self.black_model=vqa_model
        self.device=device
        self.batch=None
        self.captions=None
        self.acc_list=[]
        self.mlm_model = BertForMaskedLM.from_pretrained('bert-base-uncased', config=config_atk).to(self.device)
    def Gen_ori_feats(self, batch):
        image=batch['image'].to(self.device, non_blocking=True)
        img_feats_list,txt_feats_list = self.white_model.Gen_feats(image,batch['question'][0])
        img_feats=torch.cat(img_feats_list, axis=0)
        txt_feats=torch.cat(txt_feats_list, axis=0)
        return img_feats,txt_feats
    def pgd_attack(self,x):
        img_feats_list, txt_feats_list = self.white_model.Gen_feats(x, self.batch['question'])
        img_feats = torch.cat(img_feats_list, axis=0)
        txt_feats = torch.cat(txt_feats_list, axis=0)
        return [txt_feats,img_feats]
    def black_box_predict(self,image,text):
        answer_ids, topk_ids, topk_probs, = self.black_model(image, text,
                                                             self.answer_candidates, train=False,
                                                             inference='rank',
                                                             k_test=128)
        out_v = []
        for answer_id in answer_ids:
            out_v.append({"answer": self.answer_list[answer_id]})
        return out_v[0]['answer']
    @torch.no_grad()
    def evaluate(
        self,
        data_loader,
        tokenizer
    ):
        answer_list = data_loader.dataset.answer_list
        self.answer_list=answer_list
        answer_candidates = self.black_model.tokenizer(answer_list, padding='longest', return_tensors='pt').to(self.device)
        self.answer_candidates=answer_candidates
        answer_candidates.input_ids[:, 0] = self.black_model.tokenizer.bos_token_id
        self.tokeizer=tokenizer
        self.white_model.eval()
        self.black_model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = "Test:"
        print_freq=50000
        for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            if len(self.acc_list)>=5000:
                break
            if int(batch['question_id'][0]) not in self.correct_list:
                continue
            ori_img = batch['image'].to(self.device, non_blocking=True)
            pred_ans = self.black_box_predict(ori_img, batch['question'][0])
            ret = dict()
            ret['preds'] = [self.blip_ans_table[str(int(batch['question_id'][0]))]]
            if pred_ans!=ret['preds'][0]:
                print('wrong answer here',pred_ans,ret['preds'][0])
                continue
            self.batch = copy.deepcopy(batch)
            ori_img_feats,ori_txt_feats = self.Gen_ori_feats(batch)
            adv_img = copy.deepcopy(ori_img)
            torch.set_grad_enabled(True)
            adv_x, loss = pgd.projected_gradient_descent(self.pgd_attack, adv_img, 0.125, 0.01, self.total_stg_step,
                                                         np.inf, clip_min=-1.0,clip_max=1.0,
                                                         y=[ori_txt_feats, ori_img_feats],
                                                         time=0, ori_x=ori_img,method='BSA')
            torch.set_grad_enabled(False)
            self.attack_dict[str(int(batch['question_id'][0]))] = {'image': adv_x, 'text': batch['question'][0]}
            if len(self.attack_dict) == 10:
                for qid_key in self.attack_dict.keys():
                    adv_image = self.attack_dict[qid_key]['image']
                    adv_txt = self.attack_dict[qid_key]['text']
                    ans_after_attack = self.black_box_predict(adv_image, adv_txt)
                    if ans_after_attack != self.blip_ans_table[str(qid_key)]:
                        self.acc_list.append(1)
                    else:
                        self.acc_list.append(0)
                self.attack_dict = {}
                if len(self.acc_list) % 100 == 0 and len(self.acc_list) != 0:
                    print(f'ASR of {str(len(self.acc_list))} samples:', sum(self.acc_list) / len(self.acc_list))
        print('ASR: ', sum(self.acc_list) / len(self.acc_list))


