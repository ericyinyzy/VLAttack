
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import sys
from transformers import BertForMaskedLM, BertTokenizer
from transformers.models.bert.modeling_bert import BertConfig#, BertEmbeddings
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
        # self.label = label
        self.seq = seq_a
        self.final_adverse = seq_a
        self.query = 0
        self.change = 0
        self.success = 0
        self.sim = 0.0
        self.changes = []
import json
import cleverhans.torch.attacks.BLIP.projected_gradient_descent as pgd
class Adv_attack:
    def __init__(self, vqa_model,pretrain_model,tokenizer,device,correct_idx_list,correct_pred_list,USE_model):
        self.attack_dict = {}
        self.acc_list=[]
        self.tokenizer = tokenizer
        self.single_stg_step=20
        self.total_stg_step=40
        self.tokenizer_mlm = BertTokenizer.from_pretrained("bert-base-uncased",
                                                           do_lower_case="uncased" in "bert-base-uncased")
        self.text_bank={}
        self.adv_txt_dict = {}
        self.cos_sim=0.95
        self.k=10
        self.text_budget = 100000
        self.correct_list = correct_idx_list
        self.blip_ans_table = correct_pred_list
        self.white_model=pretrain_model
        self.black_model=vqa_model
        self.USE_model=USE_model
        self.device=device
        self.batch=None
        self.captions=None
        self.vqa_score=0
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
        word_list = []
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
    def get_important_scores(self, words,batch,tgt_pos,score,image):
        masked_words = self._get_masked(words)
        texts = [' '.join(words) for words in masked_words]
        important_scores = []
        for mlm in texts:
            # question_input = self.tokenizer(mlm, padding='longest', return_tensors="pt").to(self.device)
            _,topk_ids, topk_probs = self.black_model(image, mlm, self.answer_candidates, train=False,inference='rank', k_test=128)
            _, pred = topk_probs[0].max(dim=0)
            # print(tgt_pos)
            if tgt_pos not in list(topk_ids[0].cpu().numpy()):
                important_scores.append((torch.tensor(-10000).to(self.device)).data.cpu().numpy())
            else:
                # print(topk_probs[0], [torch.where(topk_ids[0] == tgt_pos)])
                im_value=topk_probs[0][torch.where(topk_ids[0] == tgt_pos)][0]
                important_scores.append((im_value - score).data.cpu().numpy())
        return np.array(important_scores)
    def bert_attack(self,batch,tgt_pos,score,gth):
        # self.k=10
        ori_text=batch['question'][0]
        image=batch['image']
        image = image.to(self.device, non_blocking=True)
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
        important_scores = self.get_important_scores(words,batch,tgt_pos,score,image)
        feature.query += int(len(words))
        list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=False)
        final_words = copy.deepcopy(words)
        success = 0
        simout = 1
        text_bank=[]
        sim_list=[]
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
                    answer_ids, topk_ids, topk_probs, = self.black_model(image, temp_text,
                                                                         self.answer_candidates, train=False,
                                                                         inference='rank',
                                                                         k_test=128)
                    result = []
                    for ques_id, answer_id in zip(batch['question_id'], answer_ids):
                        result.append({"question_id": int(ques_id.item()), "answer": self.answer_list[answer_id]})
                    ans_after_attack=result[0]['answer']
                    if ans_after_attack != gth:
                        success=1
                        return text_bank,success,sim_list
        text_cand=[]
        if len(text_bank)!=len(sim_list):
            print('wrong bank')
            raise ValueError
        if len(text_bank)!=0:
            sim_list_sort=copy.deepcopy(sim_list)
            for i in range(len(sim_list_sort)):
                si=sim_list_sort.index(max(sim_list_sort))
                text_cand.append(text_bank[si])
                sim_list_sort[si]=-1e8
        return text_cand,success,sim_list
    def _get_masked(self, words):
        len_text = max(len(words), 2)
        masked_words = []
        for i in range(len_text):
            masked_words.append(words[0:i] + ['[MASK]'] + words[i + 1:])
        return masked_words
    def black_box_predict(self,image,text):
        answer_ids, topk_ids, topk_probs, = self.black_model(image, text,
                                                             self.answer_candidates, train=False,
                                                             inference='rank',
                                                             k_test=128)
        out_v = []
        for  answer_id in answer_ids:
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
            self.batch = copy.deepcopy(batch)
            ori_img = batch['image'].to(self.device, non_blocking=True)
            pred_ans = self.black_box_predict(ori_img,batch['question'][0])
            ret = dict()
            ret['preds'] = [self.blip_ans_table[str(int(batch['question_id'][0]))]]
            if pred_ans!=ret['preds'][0]:
                print('wrong answer here',pred_ans,ret['preds'][0])
                continue
            ori_img_feats,ori_txt_feats = self.Gen_ori_feats(batch)
            adv_img = copy.deepcopy(ori_img)
            torch.set_grad_enabled(True)
            adv_img, loss = pgd.projected_gradient_descent(self.pgd_attack, adv_img, 0.125, 0.01, self.single_stg_step,
                                                         np.inf, clip_min=-1.0,clip_max=1.0,
                                                         y=[ori_txt_feats, ori_img_feats, None, None, None],
                                                         time=0, ori_x=ori_img,method='BSA')
            torch.set_grad_enabled(False)
            out_v_ans = self.black_box_predict(adv_img, batch['question'][0])
            if out_v_ans!=self.blip_ans_table[str(int(batch['question_id'][0]))]:
                self.acc_list.append(1)
                if len(self.acc_list) % 100 == 0 and len(self.acc_list) != 0:
                    print(f'ASR of {str(len(self.acc_list))} samples:', sum(self.acc_list) / len(self.acc_list))
                continue
            answer_ids,topk_ids, topk_probs = self.black_model(ori_img, batch['question'][0], answer_candidates, train=False,inference='rank', k_test=128)
            result = []
            for ques_id, answer_id in zip(batch['question_id'], answer_ids):
                result.append({"question_id": int(ques_id.item()), "answer": answer_list[answer_id]})
            gth=result[0]['answer']
            score, pred = topk_probs[0].max(dim=0)
            tgt_pos = topk_ids[0][pred]
            score, pred = topk_probs[0].max(dim=0)
            adv_text, success, sim = self.bert_attack(batch, tgt_pos, score,gth)
            if success==1:
                self.acc_list.append(1)
            else:
                text_bank = adv_text[:-1]
                if len(text_bank) > self.single_stg_step:
                    text_bank = text_bank[:self.single_stg_step]
                if len(text_bank) == 0:
                    text_bank = text_bank + [batch['question'][0]]
                iters = int((self.total_stg_step - self.single_stg_step) / len(text_bank))
                iters_list = [iters for _ in range(len(text_bank))]

                iters_list[-1] += self.single_stg_step - sum(iters_list)

                idx = 0
                count_iter = 0
                max_len = len(text_bank) - 1
                while True:
                    if idx > max_len:
                        break
                    else:
                        count_iter += iters
                        idx += 1
                    self.batch['question']=text_bank[idx - 1]
                    out_v_ans = self.black_box_predict(adv_img, text_bank[idx - 1])
                    if out_v_ans != self.blip_ans_table[str(int(batch['question_id'][0]))]:
                        self.acc_list.append(1)
                        break
                    torch.set_grad_enabled(True)
                    adv_img, _ = pgd.projected_gradient_descent(self.pgd_attack, adv_img, 0.125, 0.01, iters,
                                                         np.inf, clip_min=-1.0,clip_max=1.0,
                                                         y=[ori_txt_feats, ori_img_feats, None, None, None],
                                                         time=1, ori_x=ori_img,method='BSA')

                    torch.set_grad_enabled(False)



                    out_v_ans=self.black_box_predict(adv_img,text_bank[idx-1])
                    if out_v_ans != self.blip_ans_table[str(int(batch['question_id'][0]))]:
                        self.acc_list.append(1)
                        break
                if out_v_ans==self.blip_ans_table[str(int(batch['question_id'][0]))]:
                    self.acc_list.append(0)
                if len(self.acc_list) % 100 == 0 and len(self.acc_list) != 0:
                    print(f'ASR of {str(len(self.acc_list))} samples:', sum(self.acc_list) / len(self.acc_list))
        print('ASR: ', sum(self.acc_list) / len(self.acc_list))

