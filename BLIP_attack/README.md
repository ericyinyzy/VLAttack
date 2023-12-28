# VLAttack on the BLIP model

[[BLIP Paper]](https://arxiv.org/pdf/2201.12086.pdf) 

In this repository, we test VLAttack through the VQA task and NLVR task on the VQAv2 and NLVR2 datasets, respectively. 
We conducted VLAttack on 5K correctly predicted samples.
Instructions are shown below:

## Pre-trained Model Preparation
Firstly, download the pretrained BLIP model weights (BLIP with ViT-B, 14M) from the [BLIP original repository](https://github.com/salesforce/BLIP). We use these weights to generate adversarial samples in our work.

## Attack VQAv2
1. Download the VQAv2 dataset from the original [website](https://visualqa.org/download.html), and then set the `vqa_root` in `./configs/vqa.yaml` 

2. Download the finetuned VQAv2 model weights from the original repo of [BLIP](https://github.com/salesforce/BLIP). Specifically, the finetuned model weights can be downloaded from [here](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_vqa.pth). Don't forget to set the `pretrain` in `./configs/vqa.yaml` with the path of `model_vqa.pth`. 
3. Find 5K correctly predicted samples using the `python prepare_vqa.py` command. After running, it will generate `right_vqa_list.txt` and `right_vqa_ans_table.txt`, which store the indexes and predictions of correctly predicted samples.

4. To conduct VLAttack on the VQAv2 dataset, use the `python attack_vqa.py` command with different `--method` options shown below:
- **Method Options**:
  - BSA (ours)
  - VLAttack (ours)
  - [Co-Attack](https://arxiv.org/pdf/2206.09391.pdf)
  - [BERTAttack](https://arxiv.org/pdf/2004.09984.pdf)

- **Command**:
  Replace `METHOD_NAME` with your chosen options from above:
  ```bash
  python attack_vqa.py --method METHOD_NAME
  ```
## Attack NLVR2
1. Download the NLVR2 dataset from the original website, and then set the `image_root` in `./configs/nlvr.yaml` 
2. Download the finetuned NLVR2 model weights from the original repo of [BLIP](https://github.com/salesforce/BLIP). Specifically, the finetuned model weights can be downloaded from [here](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_nlvr.pth). Don't forget to set the `pretrain` in `./configs/nlvr.yaml` with the path of `model_base_nlvr.pth`. 
3. Find 5K correctly predicted samples using the `python prepare_nlvr.py` command. After running, it will generate `right_nlvr_list.txt` and `right_nlvr_ans_table.txt`, which store the indexes and predictions of correctly predicted samples.
4. To conduct VLAttack on the NLVR2 dataset, use the `python attack_nlvr.py` command with above `--method` options. For example, run below command to conduct VLAttack:
```bash
python attack_nlvr.py --method VLAttack
```


