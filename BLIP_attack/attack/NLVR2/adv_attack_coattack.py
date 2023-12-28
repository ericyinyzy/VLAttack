
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import sys
import torch.nn as nn
from transformers import BertForMaskedLM, BertTokenizer
from transformers.models.bert.modeling_bert import BertConfig
config_atk = BertConfig.from_pretrained('bert-base-uncased')
import copy
import torch
import torch.nn
import utils
import torch.optim
from filter_words import filter_words
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
filter_words = filter_words + stopwords.words('english')+['?','.']
class Feature(object):
    def __init__(self, seq_a):
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
        self.text_budget = 100000
        self.cos_sim = 0.95
        self.k = 10
        self.l2 = nn.MSELoss()
        self.correct_list = correct_idx_list
        self.blip_ans_table = correct_pred_list
        self.acc_list = []
        self.tokenizer = tokenizer
        self.tokenizer_mlm = BertTokenizer.from_pretrained("bert-base-uncased",
                                                           do_lower_case="uncased" in "bert-base-uncased")
        self.adv_txt_dict = {}
        self.white_model = copy.deepcopy(pretrain_model)
        self.black_model = copy.deepcopy(vqa_model)
        self.USE_model=USE_model
        self.device = device
        self.batch = None
        self.captions = None
        self.vqa_score = 0
        self.acc_list = []
        self.mlm_model = BertForMaskedLM.from_pretrained('bert-base-uncased', config=config_atk).to(self.device)

    def Gen_ori_feats(self, batch,feats=None):
        if feats=='feats_0':
            image=batch['image0'].to(self.device, non_blocking=True)
        elif feats=='feats_1':
            image = batch['image1'].to(self.device, non_blocking=True)
        img_feats_list, txt_feats_list = self.white_model.Gen_feats(image, batch['sentence'][0])
        img_feats = torch.cat(img_feats_list, axis=0)
        txt_feats = torch.cat(txt_feats_list, axis=0)
        return img_feats, txt_feats
    def pgd_attack(self,x):
        img_feats_list, txt_feats_list = self.white_model.Gen_feats(x, self.batch['sentence'])
        img_feats = torch.cat(img_feats_list, axis=0)
        txt_feats = torch.cat(txt_feats_list, axis=0)
        return [txt_feats[1:,:,:],img_feats]
    def _tokenize(self, seq, tokenizer):
        seq = seq.replace('\n', '').lower()
        words = seq.split(' ')
        sub_words = []
        keys = []
        index = 0
        for word in words:
            sub = tokenizer.tokenize(word)
            sub_words += sub
            keys.append([index, index + len(sub)])
            index += len(sub)
        return words, sub_words, keys
    def get_bpe_substitues(self, substitutes, tokenizer, mlm_model):
        substitutes = substitutes[0:12, 0:4]  # maximum BPE candidates
        all_substitutes = []
        for i in range(substitutes.size(0)):
            if len(all_substitutes) == 0:
                lev_i = substitutes[i]
                all_substitutes = [[int(c)] for c in lev_i]
            else:
                lev_i = []
                for all_sub in all_substitutes:
                    for j in substitutes[i]:
                        lev_i.append(all_sub + [int(j)])
                all_substitutes = lev_i
        c_loss = torch.nn.CrossEntropyLoss(reduction='none')
        all_substitutes = torch.tensor(all_substitutes)  # [ N, L ]
        all_substitutes = all_substitutes[:24].to(self.device)
        N, L = all_substitutes.size()
        word_predictions = mlm_model(all_substitutes)[0]  # N L vocab-size
        ppl = c_loss(word_predictions.view(N * L, -1), all_substitutes.view(-1))  # [ N*L ]
        ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1))  # N
        _, word_list = torch.sort(ppl)
        word_list = [all_substitutes[i] for i in word_list]
        final_words = []
        for word in word_list:
            tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
            text = tokenizer.convert_tokens_to_string(tokens)
            final_words.append(text)
        return final_words
    def get_substitues(self, substitutes, tokenizer, mlm_model, substitutes_score=None, use_bpe=True, threshold=0.3):
        words = []
        sub_len, k = substitutes.size()  # sub-len, k

        if sub_len == 0:
            return words

        elif sub_len == 1:
            for (i, j) in zip(substitutes[0], substitutes_score[0]):
                if threshold != 0 and j < threshold:
                    break
                words.append(tokenizer._convert_id_to_token(int(i)))
        else:
            if use_bpe == 1:
                words = self.get_bpe_substitues(substitutes, tokenizer, mlm_model)
            else:
                return words
        return words
    def get_important_scores(self, words,batch,tgt_feat,image0,image1):
        masked_words = self._get_masked(words)
        texts = [' '.join(words) for words in masked_words]
        important_scores = []
        for mlm in texts:
            # question_input = self.tokenizer(mlm, padding='longest', return_tensors="pt").to(self.device)
            img_feats_list, txt_feats_list0 = self.white_model.Gen_feats(image0, mlm)
            txt_feats0 = torch.cat(txt_feats_list0, axis=0)

            img_feats_list, txt_feats_list1 = self.white_model.Gen_feats(image0, mlm)
            txt_feats1 = torch.cat(txt_feats_list1, axis=0)

            src_feat0 = txt_feats0[1:, :, :]
            src_feat1 = txt_feats1[1:, :, :]
            src_feat=torch.cat([src_feat0,src_feat1],axis=0)

            feat_len = min(src_feat.shape[1], tgt_feat.shape[1])
            src_feat = src_feat[:, :feat_len]
            tgt_feat = tgt_feat[:, :feat_len]
            gap = -self.l2(src_feat, tgt_feat)
            important_scores.append((gap).data.cpu().numpy())
        return np.array(important_scores)
    def bert_attack(self,batch,tgt_feat):
        ori_text=batch['sentence'][0]
        image0=batch['image0']
        image0 = image0.to(self.device, non_blocking=True)
        image1 = batch['image1']
        image1 = image1.to(self.device, non_blocking=True)

        text=ori_text.lower()
        feature = Feature(text)
        tokenizer = self.tokenizer_mlm
        words, sub_words, keys = self._tokenize(feature.seq, tokenizer)
        max_length = 512
        inputs = tokenizer.encode_plus(feature.seq, None, add_special_tokens=True, max_length=max_length,
                                       truncation=True)
        input_ids, _ = torch.tensor(inputs["input_ids"]), torch.tensor(inputs["token_type_ids"])
        sub_words = ['[CLS]'] + sub_words[:2] + sub_words[2:max_length - 2] + ['[SEP]']
        input_ids_ = torch.tensor([tokenizer.convert_tokens_to_ids(sub_words)])
        word_predictions = self.mlm_model(input_ids_.to(self.device))[0].squeeze()  # seq-len(sub) vocab
        word_pred_scores_all, word_predictions = torch.topk(word_predictions, self.k, -1)
        word_predictions = word_predictions[1:len(sub_words) + 1, :]
        word_pred_scores_all = word_pred_scores_all[1:len(sub_words) + 1, :]
        important_scores = self.get_important_scores(words,batch,tgt_feat,image0,image1)
        feature.query += int(len(words))
        list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=False)
        final_words = copy.deepcopy(words)
        success = 0
        simout = 1
        text_bank=[]
        sim_list=[]
        most_gap = 0.0
        candidate = None
        cand_idx = None
        for ii,top_index in enumerate(list_of_index):
            if feature.change >= self.text_budget:
                feature.success = 1  # exceed
                break
            tgt_word = words[top_index[0]]
            if tgt_word in filter_words:
                continue
            if keys[top_index[0]][0] > max_length - 2:
                continue
            substitutes = word_predictions[keys[top_index[0]][0]:keys[top_index[0]][1]]
            word_pred_scores = word_pred_scores_all[keys[top_index[0]][0]:keys[top_index[0]][1]]
            substitutes = self.get_substitues(substitutes, tokenizer, self.mlm_model, substitutes_score=word_pred_scores)
            most_gap = 0.0
            candidate = None
            distance = []
            for substitute in substitutes:
                if substitute == tgt_word:
                    continue  # filter out original word
                if '##' in substitute:
                    continue  # filter out sub-word
                if substitute in filter_words:
                    continue
                temp_replace = copy.deepcopy(final_words)
                temp_replace[top_index[0]] = substitute
                temp_text = tokenizer.convert_tokens_to_string(temp_replace)
                embs=self.USE_model([ori_text,temp_text]).numpy()
                norm = np.linalg.norm(embs, axis=1)
                embs = embs / norm[:, None]
                sim = (embs[:1] * embs[1:]).sum(axis=1)[0]
                if sim>self.cos_sim:
                    sim_list.append(sim)
                    text_bank.append(temp_text)

                    img_feats_list, txt_feats_list0 = self.white_model.Gen_feats(image0, temp_text)
                    txt_feats0 = torch.cat(txt_feats_list0, axis=0)
                    img_feats_list, txt_feats_list1 = self.white_model.Gen_feats(image1, temp_text)
                    txt_feats1 = torch.cat(txt_feats_list1, axis=0)
                    src_feat0 = txt_feats0[1:, :, :]
                    src_feat1 = txt_feats1[1:, :, :]
                    src_feat=torch.cat([src_feat0,src_feat1],axis=0)
                    feat_len = min(src_feat.shape[1], tgt_feat.shape[1])
                    src_feat = src_feat[:,:feat_len]
                    tgt_feat = tgt_feat[:,:feat_len]
                    gap = self.l2(src_feat, tgt_feat)
                    if gap > most_gap:
                        most_gap = gap
                        candidate = substitute
                        cand_idx = top_index[0]
        if most_gap > 0:
            final_words[cand_idx] = candidate
        return tokenizer.convert_tokens_to_string(final_words)
    def _get_masked(self, words):
        len_text = max(len(words), 2)
        masked_words = []
        for i in range(len_text):
            masked_words.append(words[0:i] + ['[MASK]'] + words[i + 1:])
        return masked_words
    @torch.no_grad()
    def evaluate(
        self,
        data_loader,
        tokenizer
    ):
        self.tokeizer = tokenizer
        self.white_model.eval()
        self.black_model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = "Test:"
        print_freq=50000
        index = 0
        for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            index += 1
            if len(self.acc_list)>=5000:
                break
            if int(index) not in self.correct_list:
                continue
            ori_img_I,ori_img_II=batch['image0'].to(self.device),batch['image1'].to(self.device)
            images = torch.cat([ori_img_I, ori_img_II], dim=0)
            targets =  batch['label'].to(self.device)
            prediction = self.black_model(images, batch['sentence'], targets=targets, train=False)
            _, pred_class = prediction.max(1)
            self.batch = copy.deepcopy(batch)
            ori_img_feats_I, ori_txt_feats_I = self.Gen_ori_feats(batch, feats='feats_0')
            ori_img_feats_II, ori_txt_feats_II = self.Gen_ori_feats(batch, feats='feats_1')
            adv_img_I = copy.deepcopy(ori_img_I)
            adv_img_II = copy.deepcopy(ori_img_II)

            tgt_feats_I = ori_txt_feats_I[1:, :, :]
            tgt_feats_II=ori_txt_feats_II[1:, :, :]

            tgt_feats=torch.cat([tgt_feats_I,tgt_feats_II],axis=0)
            adv_text = self.bert_attack(batch, tgt_feats)


            question_input = self.tokenizer_mlm(
                adv_text,
                padding='longest',
                truncation=True,
                max_length=35,
                return_tensors="pt"
            ).to(self.device)
            self.batch[f"text_ids"] = question_input["input_ids"].cuda()
            self.batch[f"text_masks"] = question_input["attention_mask"].cuda()
            img_feats_list, txt_feats_list = self.white_model.Gen_feats(ori_img_I, adv_text)
            txt_feats = torch.cat(txt_feats_list, axis=0)
            adv_tgt_feats_I = txt_feats[1:, :, :]

            question_input = self.tokenizer_mlm(
                adv_text,
                padding='longest',
                truncation=True,
                max_length=35,
                return_tensors="pt"
            ).to(self.device)
            self.batch[f"text_ids"] = question_input["input_ids"].cuda()
            self.batch[f"text_masks"] = question_input["attention_mask"].cuda()
            img_feats_list, txt_feats_list = self.white_model.Gen_feats(ori_img_II, adv_text)
            txt_feats = torch.cat(txt_feats_list, axis=0)
            adv_tgt_feats_II = txt_feats[1:, :, :]

            torch.set_grad_enabled(True)
            adv_img_I, _ = pgd.projected_gradient_descent(self.pgd_attack, adv_img_I, 0.125, 0.01, 40,
                                                        np.inf, clip_min=-1.0, clip_max=1.0,
                                                        y=[tgt_feats_I,adv_tgt_feats_I],
                                                        time=0, ori_x=ori_img_I,method='Co-Attack')
            adv_img_II, _ = pgd.projected_gradient_descent(self.pgd_attack, adv_img_II, 0.125, 0.01, 40,
                                                         np.inf, clip_min=-1.0, clip_max=1.0,
                                                         y=[tgt_feats_II, adv_tgt_feats_II],
                                                         time=0, ori_x=ori_img_II,method='Co-Attack')
            torch.set_grad_enabled(False)


            adv_imgs = torch.cat([adv_img_I, adv_img_II], dim=0)
            adv_imgs, targets = adv_imgs.to(self.device), batch['label'].to(self.device)
            prediction = self.black_model(adv_imgs, batch['sentence'], targets=targets, train=False)
            _, pred_class = prediction.max(1)
            if pred_class != targets:
                self.acc_list.append(1)
            else:
                self.acc_list.append(0)
            if len(self.acc_list) % 100 == 0 and len(self.acc_list) != 0:
                print(f'ASR of {str(len(self.acc_list))} samples:', sum(self.acc_list) / len(self.acc_list))
        print('ASR: ', sum(self.acc_list) / len(self.acc_list))
