import numpy as np
import sys
from transformers import BertForMaskedLM, BertTokenizer
from transformers.models.bert.modeling_bert import BertConfig#, BertEmbeddings
config_atk = BertConfig.from_pretrained('bert-base-uncased')
import copy
import torch
import torch.nn
import utils
import torch.optim
import cleverhans.torch.attacks.BLIP.projected_gradient_descent as pgd
# exit()
class Adv_attack:
    def __init__(self, nlvr_model,pretrain_model,tokenizer,device,correct_idx_list,correct_pred_list,USE_model):
        self.attack_dict = {}
        self.acc_list=[]
        self.tokenizer = tokenizer
        self.tokenizer_mlm = BertTokenizer.from_pretrained("bert-base-uncased",
                                                           do_lower_case="uncased" in "bert-base-uncased")
        self.total_stg_step=40
        self.correct_list = correct_idx_list
        self.blip_ans_table = correct_pred_list
        self.adv_txt_dict = {}
        self.white_model=pretrain_model
        self.black_model=nlvr_model
        self.device=device
        self.batch=None
        self.captions=None
        self.vqa_score=0
        self.acc_list=[]
        self.mlm_model = BertForMaskedLM.from_pretrained('bert-base-uncased', config=config_atk).to(self.device)

    def Gen_ori_feats(self, batch,feats=None):
        if feats=='feats_0':
            image=batch['image0'].to(self.device, non_blocking=True)
        elif feats=='feats_1':
            image = batch['image1'].to(self.device, non_blocking=True)
        img_feats_list,txt_feats_list = self.white_model.Gen_feats(image,batch['sentence'][0])
        img_feats=torch.cat(img_feats_list, axis=0)
        txt_feats=torch.cat(txt_feats_list, axis=0)
        return img_feats,txt_feats
    def pgd_attack(self,x):
        img_feats_list, txt_feats_list = self.white_model.Gen_feats(x, self.batch['sentence'])
        img_feats = torch.cat(img_feats_list, axis=0)
        txt_feats = torch.cat(txt_feats_list, axis=0)
        return [txt_feats,img_feats]
    def black_box_predict(self,adv_img_I,adv_img_II):
        adv_imgs = torch.cat([adv_img_I, adv_img_II], dim=0)
        prediction = self.black_model(adv_imgs, self.batch['sentence'], targets=self.batch['label'].to(self.device), train=False)
        _, pred_class = prediction.max(1)
        return pred_class
    @torch.no_grad()
    def evaluate(
        self,
        data_loader,
        tokenizer
    ):
        self.tokeizer=tokenizer
        self.white_model.eval()
        self.black_model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = "Test:"
        print_freq=50000
        index=0
        for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            index += 1
            if len(self.acc_list)>=5000:
                break
            if int(index) not in self.correct_list:
                continue
            ori_img_I, ori_img_II = batch['image0'].to(self.device), batch['image1'].to(self.device)
            self.ori_imgs = torch.cat([ori_img_I, ori_img_II], dim=0)
            targets = batch['label'].to(self.device)
            self.batch = copy.deepcopy(batch)
            prediction = self.black_model(self.ori_imgs, batch['sentence'], targets=targets, train=False)
            _, pred_class = prediction.max(1)
            if targets != pred_class:
                continue


            ori_img_feats_I,ori_txt_feats_I = self.Gen_ori_feats(batch,feats='feats_0')
            ori_img_feats_II, ori_txt_feats_II = self.Gen_ori_feats(batch,feats='feats_1')
            adv_img_I = copy.deepcopy(ori_img_I)
            adv_img_II = copy.deepcopy(ori_img_II)
            torch.set_grad_enabled(True)
            adv_img_I, _ = pgd.projected_gradient_descent(self.pgd_attack, adv_img_I, 0.125, 0.01, self.total_stg_step,
                                                         np.inf, clip_min=-1.0,clip_max=1.0,
                                                         y=[ori_txt_feats_I, ori_img_feats_I],
                                                         time=0, ori_x=ori_img_I,method='BSA')
            adv_img_II, _ = pgd.projected_gradient_descent(self.pgd_attack, adv_img_II, 0.125, 0.01, self.total_stg_step,
                                                        np.inf, clip_min=-1.0, clip_max=1.0,
                                                        y=[ori_txt_feats_II, ori_img_feats_II],
                                                        time=0, ori_x=ori_img_II,method='BSA')
            torch.set_grad_enabled(False)
            attack_dict = {'image0': adv_img_I,'image1': adv_img_II, 'text': batch['sentence'][0]}
            self.adv_txt_dict[str(int(index))] = batch['sentence'][0]
            self.attack_dict[str(int(index))] = attack_dict
            if len(self.attack_dict) == 10:
                for qid_key in self.attack_dict.keys():
                    adv_image_I = self.attack_dict[qid_key]['image0']
                    adv_image_II = self.attack_dict[qid_key]['image1']
                    self.batch['sentence'] = self.attack_dict[qid_key]['text']
                    ans_after_attack=self.black_box_predict(adv_image_I,adv_image_II)
                    if self.blip_ans_table[str(int(qid_key))] != int(ans_after_attack.detach().cpu().numpy()[0]):
                        self.acc_list.append(1)
                    else:
                        self.acc_list.append(0)
                self.attack_dict = {}
                if len(self.acc_list) % 100 == 0 and len(self.acc_list) != 0:
                    print(f'ASR of {str(len(self.acc_list))} samples:', sum(self.acc_list) / len(self.acc_list))
        print('ASR: ', sum(self.acc_list) / len(self.acc_list))


