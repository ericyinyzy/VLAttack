import copy
from collections import defaultdict
from pathlib import Path
import random
import json

import torch
import torch.utils.data
from transformers import RobertaTokenizerFast

import util.dist as dist
from util.box_ops import generalized_box_iou

import sys
sys.path.append("..")
from models.unitab import target2prevind as target2prevind_caption
from models.unitab import target2gtind as target2gtind_caption

from .coco import ModulatedDetection, make_coco_transforms

class RefExpDetection(ModulatedDetection):
    def __init__(self, img_folder, ann_file, transforms, return_tokens, tokenizer, is_train=False,\
        max_decoding_step=256, num_queries=200):
        super(RefExpDetection, self).__init__(img_folder, ann_file, transforms, False, return_tokens, tokenizer)
        self.tokenizer = tokenizer
        self.max_decoding_step = max_decoding_step
        self.num_queries = num_queries

    def __getitem__(self, idx):
        img, target = super(RefExpDetection, self).__getitem__(idx)

        ## output_text with bbox version
        tokenized = self.tokenizer.batch_encode_plus([target["caption"]], padding="max_length", \
            max_length=self.max_decoding_step, truncation=True, return_tensors="pt")
        target_gt = target2gtind_caption(tokenized['input_ids'], \
            [target],num_bins=self.num_queries)
        previdx_gt = target2prevind_caption(tokenized['input_ids'], \
            [target],num_bins=self.num_queries)
        target['target_gt'] = target_gt
        target['previdx_gt'] = previdx_gt
        target['sentence_id'] = torch.ones(1).long()*-1
        target['original_img_id'] = torch.ones(1).long()*-1

        # exit()
        return img, target

# Main RefCOCO
class RefExpEvaluator(object):
    def __init__(self, refexp_gt, iou_types, k=[1], thresh_iou=0.5):
        assert isinstance(k, (list, tuple))
        refexp_gt = copy.deepcopy(refexp_gt)
        self.refexp_gt = refexp_gt
        self.iou_types = iou_types
        self.img_ids = self.refexp_gt.imgs.keys()
        self.predictions = {}
        self.img_list={}
        self.correct_list=[]
        self.all_list=[]
        self.k = k
        self.thresh_iou = thresh_iou

    def accumulate(self):
        pass

    def update(self, predictions):
        self.predictions.update(predictions)
        self.img_list=predictions

    def synchronize_between_processes(self):
        all_predictions = dist.all_gather(self.predictions)
        merged_predictions = {}
        for p in all_predictions:
            merged_predictions.update(p)
        self.predictions = merged_predictions

    def summarize(self):
        # print('img_ids',self.predictions.keys())
        # exit()
        success=1
        if dist.is_main_process():
            dataset2score = {
                "refcoco": {k: 0.0 for k in self.k},
                "refcoco+": {k: 0.0 for k in self.k},
                "refcocog": {k: 0.0 for k in self.k},
            }
            dataset2count = {"refcoco": 0.0, "refcoco+": 0.0, "refcocog": 0.0}
            for image_id in self.img_list.keys():
                ann_ids = self.refexp_gt.getAnnIds(imgIds=image_id)
                assert len(ann_ids) == 1
                img_info = self.refexp_gt.loadImgs(image_id)[0]

                target = self.refexp_gt.loadAnns(ann_ids[0])
                # print('target',target)
                prediction = self.predictions[image_id]

                assert prediction is not None
                sorted_scores_boxes = sorted(
                    zip(prediction["scores"].tolist(), prediction["boxes"].tolist()), reverse=True
                )
                # print('sorted_boxes',sorted_scores_boxes)
                sorted_scores, sorted_boxes = zip(*sorted_scores_boxes)
                sorted_boxes = torch.cat([torch.as_tensor(x).view(1, 4) for x in sorted_boxes])
                # print('sorted_boxes',sorted_boxes)
                target_bbox = target[0]["bbox"]
                converted_bbox = [
                    target_bbox[0],
                    target_bbox[1],
                    target_bbox[2] + target_bbox[0],
                    target_bbox[3] + target_bbox[1],
                ]
                # print('converted',converted_bbox)
                # print(converted_bbox[0])
                # exit()
                giou = generalized_box_iou(sorted_boxes, torch.as_tensor(converted_bbox).view(-1, 4))
                # print(type(giou),giou[0][0].data.cpu())
                # exit()

                # print(self.thresh_iou,max(giou[:1]))
                # print('pred',sorted_boxes)

                # exit()
                for k in self.k:
                    if max(giou[:k]) >= self.thresh_iou:
                        dataset2score[img_info["dataset_name"]][k] += 1.0
                        self.correct_list.append(str(image_id))
                        success=0
                dataset2count[img_info["dataset_name"]] += 1.0
                self.all_list.append(str(image_id))

            for key, value in dataset2score.items():
                for k in self.k:
                    try:
                        value[k] /= dataset2count[key]
                    except:
                        pass
            results = {}
            for key, value in dataset2score.items():
                results[key] = sorted([v for k, v in value.items()])
                # print(f" Dataset: {key} - Precision @ 1, 5, 10: {results[key]} \n")

            return results,success,sorted_boxes
        return None

def build(image_set, args):
    img_dir = Path(args.coco_path) / "train2014"

    refexp_dataset_name = args.refexp_dataset_name
    print(args.refexp_dataset_name)
    print('name',refexp_dataset_name, args.test_type)
    if refexp_dataset_name in ["refcoco", "refcoco+", "refcocog"]:
        if args.test:
            test_set = args.test_type
            ann_file = Path(args.refexp_ann_path) / f"finetune_{refexp_dataset_name}_{test_set}.json"
        else:
            ann_file = Path(args.refexp_ann_path) / f"finetune_{refexp_dataset_name}_{image_set}.json"
    elif refexp_dataset_name in ["all"]:
        ann_file = Path(args.refexp_ann_path) / f"final_refexp_{image_set}.json"
    else:
        assert False, f"{refexp_dataset_name} not a valid datasset name for refexp"

    tokenizer = RobertaTokenizerFast.from_pretrained(args.text_encoder_type)
    dataset = RefExpDetection(
        img_dir,
        ann_file,
        transforms=make_coco_transforms(image_set, cautious=True),
        return_tokens=True,
        tokenizer=tokenizer,
        is_train=image_set=="train",
        max_decoding_step=args.max_decoding_step,
        num_queries=args.num_queries,
    )
    return dataset
