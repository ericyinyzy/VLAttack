import clip
import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from PIL import Image
import copy
import os
import sys
import importlib
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append('../cleverhans')

test_transform_CLIP = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
class SVHNCUSTOM(Dataset):

    def __init__(self, numpy_file, class_type, transform=None):
        """
        Args:
            numpy_file (string): Path to the numpy file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_array = np.load(numpy_file)
        self.data = self.input_array['x']
        self.targets = self.input_array['y'][:,0].tolist()
        self.classes = class_type
        self.transform = transform
    def __len__(self):
        return self.data.shape[0]
class SVHN10Mem(SVHNCUSTOM):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return index,img, target
classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
print('load_test_success')


def find_clip_right(test_loader):
    correct = 0
    correct_index = []
    with torch.no_grad():
        for index,images, labels in test_loader:
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            logits,_ = black_model(images)
            pred = logits.argmax(dim=1, keepdim=True)
            success=pred.eq(labels.view_as(pred)).sum().item()
            if success==1:
                correct_index.append(str(index))
            if len(correct_index)==5000:
                break
            correct += pred.eq(labels.view_as(pred)).sum().item()
        test_acc = 100. * correct / index
    print('test_acc',test_acc)
    return correct_index


def pgd_attack(x):
    _, feats_list = white_model(x)
    return [feats_list]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', default='/data/ziyi/svhn_2/test_224.npz')
    parser.add_argument('--model', default='ViT-B/16')
    parser.add_argument('--method', default='BSA')
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_data_clean = SVHN10Mem(numpy_file=args.test_path, class_type=classes,
                                transform=test_transform_CLIP)

    test_loader = DataLoader(test_data_clean, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)
    if args.model=='RN50':
        model, preprocess = clip.load("RN50", device=device)
        model_path="clip_model/finetuned_svhn_RN50.pth"
        print('Load RN50 success!')
    elif args.model=='ViT-B/16':
        model, preprocess = clip.load("ViT-B/16", device=device)
        model_path="clip_model/finetuned_svhn_ViTB16.pth"
        print('Load ViT-B/16 success!')
    else:
        print('only support RN50 and ViTB/16 now!')
        raise ValueError
    if args.method=='BSA' or args.method=='SSP' or args.method=='DR':
        import cleverhans.torch.attacks.CLIP.projected_gradient_descent as pgd
    elif args.method=='FDA':
        import cleverhans.torch.attacks.CLIP.projected_gradient_descent_FDA as pgd
    else:
        print('attack method is not supported!')
        raise ValueError
    white_model = copy.deepcopy(model)
    model.load_state_dict(torch.load(model_path))
    black_model = model
    black_model.eval()
    model = model.to(device)
    cnt = 0
    correct=0
    clip_right_list=find_clip_right(test_loader)
    with torch.no_grad():
        for index, images, labels in tqdm(test_loader):
            if str(index) in clip_right_list:
                cnt += 1
                images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                _, feats_list = white_model(images)
                adv_img = copy.deepcopy(images)
                torch.set_grad_enabled(True)
                adv_x, _ = pgd.projected_gradient_descent(pgd_attack, adv_img, 0.125, 0.01, 40, np.inf, clip_min=-1,
                                                          clip_max=1, y=[feats_list], time=0,
                                                          ori_x=images,method=args.method,model=args.model)

                torch.set_grad_enabled(False)
                logits, _ = black_model(adv_x)
                pred = logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
                if cnt % 100 == 0:
                    print(1 - correct / cnt)
    print(1 - correct / cnt)
