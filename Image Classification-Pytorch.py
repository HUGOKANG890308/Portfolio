#https://www.kaggle.com/code/abhishekv5055/image-classification-pytorch-90-accuracy

import  os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import torchvision.transforms as tt
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
import time
start = time.time()
os.chdir("C:\\Users\\User\\OneDrive\\深度學習課\\期末報告")
#os.chdir("C:\\Users\\User\\Desktop")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
'''
TRAIN_PATH  = 'data\\seg_train\\seg_train'
TEST_PATH=  'data\\seg_test\\seg_test'
PRED_PATH ='data\\seg_pred\\seg_pred'
'''
TRAIN_PATH  = 'data\\seg_train'
TEST_PATH=  'data\\seg_test'
PRED_PATH ='data\\seg_pred' 

transform = tt.Compose([
    tt.ToTensor(),
    tt.Resize((64, 64)),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
])
train_ds = ImageFolder(TRAIN_PATH, transform=transform)
test_ds = ImageFolder(TEST_PATH, transform=transform)
image, label = train_ds[0]
classes = train_ds.classes
#檢查image size
print(f"Image Size: {image.shape}")
batch_size=32

#test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=24)
whole_set=datasets.ImageFolder(TRAIN_PATH,transform=transform)
len1 = len(whole_set)
train_size,validate_size=int(0.8*len1),int(0.2*len1)+1 #不能整除把餘數加到後面
train_set,validate_set=torch.utils.data.random_split(whole_set,[train_size,validate_size])
train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=24)
vaild_dl= torch.utils.data.DataLoader(dataset =validate_set, batch_size = batch_size,pin_memory=True, num_workers=24)
test_dl  = torch.utils.data.DataLoader(dataset=test_ds, batch_size=batch_size,pin_memory=True, num_workers=24)
#秀出第一個batch 的圖片
'''
for batch in train_dl:
    plt.figure(figsize=(16, 8))
    image, _ = batch
    plt.imshow(make_grid(image, nrow=16).permute(1, 2, 0))
    plt.axis("off")
    plt.show()
    break'''
class IntelCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3, 32, kernel_size=3, padding=1)        
        self.conv2=nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3=nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2)
        self.flatten=nn.Flatten()
        self.fc1=nn.Linear(64*32*32, 512)
        self.fc2=nn.Linear(512, 64)
        self.fc3=nn.Linear(64, 6)
        self.dropout=nn.Dropout(0.25)
    
    def forward(self, x):
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x)))
        x=self.flatten(x)
        x=F.relu(self.fc1(x))
        x=self.dropout(x)
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        return x

model = IntelCNN()
def accuracy(pred, label):
    _, out = torch.max(pred, dim=1)
    return torch.tensor(torch.sum(out == label).item()/len(pred))

def validation_step(valid_dl, model, loss_fn):
    for image, label in valid_dl:
        if device !='cpu':
            image=image.cuda()
            label=label.cuda() 
        out = model(image)
        loss = loss_fn(out, label)
        acc = accuracy(out, label)
        return {"val_loss": loss, "val_acc": acc}

def fit(train_dl, valid_dl,test_dl, epochs, optimizer, loss_fn, model):
    history = []
    for epoch in (epochs):
        for image, label in train_dl:
            if device !='cpu':
                image=image.cuda()
                label=label.cuda() 
            out = model(image)
            loss = loss_fn(out, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        val = validation_step(valid_dl, model, loss_fn)
        tes=validation_step(test_dl, model, loss_fn)
        print(f"Epoch [{epoch}/{epochs}] => loss: {loss}, val_loss: {val['val_loss']}, val_acc: {val['val_acc']},test_acc: {tes['val_acc']}")
        history.append({"loss": loss, 
                        "val_loss": val['val_loss'], 
                        "val_acc": val['val_acc'],"test_acc": tes['val_acc']
                       })
    return history
def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    
    def __iter__(self):
        for x in self.dl:
            yield to_device(x, self.device)
model = model.to(device)
#呼叫模型參數數量
#numl=[p.numel()for p in model.parameters()]
train_dl = DeviceDataLoader(train_dl, device)
vaild_dl = DeviceDataLoader(vaild_dl, device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
epochs = np.arange(1,100,10)
history = fit(train_dl, vaild_dl,test_dl, epochs, optimizer, loss_fn, model)

train_loss = [x['loss'] for x in history]
val_loss = [x['val_loss'] for x in history]
val_acc = [x['val_acc'] for x in history]
test_acc = [x['test_acc'] for x in history]

train_loss = [x.item() for x in train_loss]
val_loss = [x.item() for x in val_loss]
val_acc = [x.item() for x in val_acc]
test_acc  = [x.item() for x in test_acc ]
epoch = epochs
plt.figure(figsize=(6,4))
plt.plot(epoch, train_loss)
plt.plot(epoch, val_loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend(['train', 'val'],loc="lower right");

epoch = epochs
plt.figure(figsize=(6,4))
plt.plot(epoch, val_acc,label='val_acc')
plt.plot(epoch, test_acc,label='test_acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc="lower right")
plt.title('Accuracy of each epochs');

def predict(model, batch_size=32, device=device, dataloader=test_dl):
    classes = ('buildings', 'forest', 'glacier', 'mountain',
           'sea', 'street')
    
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(6)]
        n_class_samples = [0 for i in range(6)]
        for images, labels in test_dl:
            if device !='cpu':
                images=images.cuda()
                labels=labels.cuda()    
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
            #print(labels.size())
            for i in range(labels.size(dim=0)):
                try:
                    label = labels[i]
                    pred = predicted[i]
                    if (label == pred):
                        n_class_correct[label] += 1
                    n_class_samples[label] += 1
                except : break

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network: {acc} %')

        for i in range(6):
            if  n_class_samples[i]!=0:                
                acc = 100.0 * n_class_correct[i] / n_class_samples[i]
                print(f'Accuracy of_test {classes[i]}: {acc} %')  
            else:
                print(f'sample_test of {classes[i]}=0')
                continue
    
predict(model)   
end = time.time()
print('程式運行時間',end - start)    
model_file = 'intel image model8.mdl'
torch.save(model, model_file)

