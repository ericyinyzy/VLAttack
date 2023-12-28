import clip
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from torch.optim import lr_scheduler
from PIL import Image
import os
import argparse
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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
class SVHNMem(SVHNCUSTOM):
    """SVHN Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, target
classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='/data/ziyi/svhn_2/train_224.npz')
    parser.add_argument('--test_path', default='/data/ziyi/svhn_2/test_224.npz')
    parser.add_argument('--model', default='RN50')
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.model=='RN50':
        model, preprocess = clip.load("RN50", device=device)
        model_name = 'RN50'
    elif args.model=='ViT-B/16':
        model, preprocess = clip.load("ViT-B/16", device=device)
        model_name='ViTB16'
    else:
        raise ValueError('only support RN50 and ViT-B/16 now!')
    model = model.to(device)
    print('load_train_success')
    train_data = SVHNMem(numpy_file=args.train_path, class_type=classes, transform=test_transform_CLIP)
    print('load_test_success')
    test_data_clean = SVHNMem(numpy_file=args.test_path, class_type=classes, transform=test_transform_CLIP)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_data_clean, batch_size=64, shuffle=False, num_workers=16,pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler1 = lr_scheduler.StepLR(optimizer, 2, 0.5)
    criterion = torch.nn.CrossEntropyLoss()
    num_epochs=10
    overall_loss=0.0
    correct=0
    iter=0
    for epoch in range(num_epochs):
        model.train()
        if epoch%3==0 and epoch!=0:
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["lr"]/2
        print('epoch:',epoch)

        for param_group in optimizer.param_groups:
            print(param_group["lr"])
        for images, labels in tqdm(train_loader):
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            optimizer.zero_grad()
            logits,_ = model(images)
            loss = criterion(logits, labels.long())
            loss.backward()
            optimizer.step()
            if iter % 100 == 0:
                print('iter: ',iter, 'loss: ', loss)
            iter += 1
            overall_loss += loss.item()
        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch,overall_loss * train_loader.batch_size / len(train_loader.dataset)))
    if not os.path.exists('clip_model'):
        os.makedirs('clip_model')
    torch.save(model.state_dict(), f'clip_model/finetuned_svhn_{model_name}.pth')
    correct=0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            logits,_ = model(images)
            loss = criterion(logits, labels.long())
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
        test_acc = 100. * correct / len(test_loader.dataset)
    print('{{"accuracy": "value": {}}}'.format(
        100. * correct / len(test_loader.dataset)))
