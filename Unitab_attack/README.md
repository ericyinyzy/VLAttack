**VLAttack**: Multimodal Adversarial Attacks on Vision Language Tasks via Pretrained Models
=======

Containing codes on Refcoco dataset on Unitab model. Codes are modified based on 
[Unitab](https://github.com/microsoft/UniTAB), [cleverhans](https://github.com/cleverhans-lab/cleverhans) and [Openattack](https://github.com/thunlp/OpenAttack)

## Data & Model Prepare

Download RefCOCO datatset from the seperqte fine-tuning model on RefCOCO and pretrained model in
[Unitab](https://github.com/microsoft/UniTAB).

Modify the `configs/refcoco.json` in `UniTAB_ATTACK` and `UniTAB_bert_ATTACK`

## Set Envirionment

```
conda create -n unitab_att python=3.8
conda activate unitab_att
```
separately install [numpy](https://pypi.org/project/numpy/) and [pytorch==1.8.1](https://pytorch.org/get-started/previous-versions/).

Then

```
pip install -r requirements.txt
```

## Evaluation

* in  `UniTAB_bert_ATTACK`, run
```
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main.py --load_white white_box_path --load_black black_box_path --dataset_config configs/refcoco.json --ema --eval --test
```
`white_box_path` is the path of the pretrained model. `black_box_path` is the path of the seperqte fine-tuning model on RefCOCO.

Run above codes to generate perturbed text list saved in `all_adv_bert_attack_bank_on_refcoco_ori_90.txt`.


move perturbed text list into `UniTAB_ATTACK` and run :
```
mv all_adv_bert_attack_bank_on_refcoco_ori_90.txt ../UniTAB_ATTACK
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main.py --load_white white_box_path --load_black black_box_path --dataset_config configs/refcoco.json --ema --eval --test
```
to generate ASR on RefCOCO dataset on Unitab model. Or you can directly run
```
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main.py --load_white white_box_path --load_black black_box_path --dataset_config configs/refcoco.json --ema --eval --test
```
in `UniTAB_ATTACK` as we have prepared `all_adv_bert_attack_bank_on_refcoco_ori_90.txt` in advance in `UniTAB_ATTACK`.
