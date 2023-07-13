import csv
from pickletools import uint8
from turtle import forward
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision
import os
import pandas as pd
import json
from PIL import Image


class MyModel(nn.Module) :
    def __init__(self):
        super(MyModel, self).__init__()
        #TODO: Make your own model
        self.block = self.blocks()
        self.linear = self.linears()
        #print(type(self.block))

    def forward(self,x) :
        #TODO:
        x = self.block(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
        
    def blocks(self):
        block = []
        #layer1
        block.extend([nn.Conv2d(3,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.ReLU()])
        block.extend([nn.Conv2d(64,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.ReLU()])
        block.extend([nn.MaxPool2d(2,stride=2)])
        #layer2
        block.extend([nn.Conv2d(64,128,kernel_size=3,padding=1),nn.BatchNorm2d(128),nn.ReLU()])
        block.extend([nn.Conv2d(128,128,kernel_size=3,padding=1),nn.BatchNorm2d(128),nn.ReLU()])
        block.extend([nn.MaxPool2d(2,stride=2)])
        #layer3
        block.extend([nn.Conv2d(128,256,kernel_size=3,padding=1),nn.BatchNorm2d(256),nn.ReLU()])
        block.extend([nn.Conv2d(256,256,kernel_size=3,padding=1),nn.BatchNorm2d(256),nn.ReLU()])
        block.extend([nn.MaxPool2d(2,stride=2)])
        #layer4
        block.extend([nn.Conv2d(256,512,kernel_size=3,padding=1),nn.BatchNorm2d(512),nn.ReLU()])
        block.extend([nn.Conv2d(512,512,kernel_size=3,padding=1),nn.BatchNorm2d(512),nn.ReLU()])
        block.extend([nn.MaxPool2d(2,stride=2)])
        
        #layer5
        block.extend([nn.Conv2d(512,512,kernel_size=3,padding=1),nn.BatchNorm2d(512),nn.ReLU()])
        block.extend([nn.Conv2d(512,512,kernel_size=3,padding=1),nn.BatchNorm2d(512),nn.ReLU()])
        block.extend([nn.MaxPool2d(2,stride=2)])
        
        return nn.Sequential(*block)

    
    def linears(self):
        block = []

        #layer6 ()
        block.extend([nn.Linear(25088,4096),nn.ReLU(),nn.Linear(4096,4096),nn.ReLU(),nn.Linear(4096,1000),nn.ReLU(),nn.Linear(1000,80)])
        return nn.Sequential(*block)


class MyDataset(Dataset) :

    def __init__(self,meta_path,root_dir,transform=None) :
        super(MyDataset,self).__init__()
        
        with open(meta_path,"r") as answer:
            self.labels = json.load(answer)
        self.classes = self.labels.get('categories')
        self.labels = self.labels.get('annotations')

        
       # print(self.classes)
    
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self) :
        return len(self.labels)
    
    def __getitem__(self,idx) :
        ipath = self.root_dir +"/"+ self.labels[idx].get('file_name')
        #print(ipath)
        image = Image.open(ipath)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        #print(image)
        label = [int(self.labels[idx].get('category')),self.labels[idx].get('file_name')]
      
        

        image= self.transform(image)
            
        return image,label

def jsonmaker(path):
  
  data = {"annotations": [] , "categories": []}

  for i in os.listdir(path):
    data['annotations'].append({"file_name": i,"category": 0})
  with open('./tojson.json', 'w', encoding='utf-8') as file:
    json.dump(data, file)


def train() :
    #TODO: Make your own training code

    # You SHOULD save your model by
    # torch.save(model.state_dict(), './checkpoint.pth') 
    # You SHOULD not modify the save path
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    train_transform = transforms.Compose([
    #transforms.ToPILImage(),
    transforms.RandomResizedCrop(224),
    transforms.ColorJitter(hue=0.1,contrast=0.1,saturation=0.1,),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
    ])
    

    train_dataset = MyDataset('./answer.json','./train_data',transform=train_transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)
    model =  MyModel()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.005)
    classes = train_dataset.classes
    
    
    for epoch in range(1):
        
        running_loss= 0
        for i,data in enumerate(train_loader,0):
            image, label = data
            image = image.to(device)
            #label = list(map(int, label))
            label = label[0]
            label = torch.tensor(label)
            label = label.to(device)
            optimizer.zero_grad()

            output = model(image)
            loss = criterion(output,label)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d,%5d] loss: %.3f' %
                      (epoch+1, i+1,running_loss/100))
                running_loss = 0
    torch.save(model.state_dict(), './model.pth') 

def get_model(model_name, checkpoint_path):
    
    model = model_name()
    model.load_state_dict(torch.load(checkpoint_path))
    
    return model


def test():
    model_name = MyModel
    checkpoint_path = './model.pth' 
    mode = 'test' 
    data_dir = "./test_data"
    meta_path = "./answer.json"
    model = get_model(model_name,checkpoint_path)

    data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomResizedCrop(224)
    ])
    jsonmaker(data_dir)
    keydata = './tojson.json'
    # Create training and validation datasets
    test_datasets = MyDataset(keydata, data_dir, data_transforms)

    # Create training and validation dataloaders
    test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=64, shuffle=False, num_workers=4)

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    model = model.to(device)

    # Set model as evaluation mode
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    

    # Inference
    result = []
    for filename,data in enumerate(test_dataloader,0):
        images, label = data
        num_image = images.shape[0]
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        for i in range(num_image):
            result.append({
                'filename': label[1][i],
                'class': preds[i].item()
            })


    result = sorted(result,key=lambda x : int(x['filename'].split('.')[0]))
    
    # Save to csv
    with open('./result.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['filename','class'])
        for res in result:
            writer.writerow([res['filename'], res['class']])


def main() :
    train()
    test()
  

if __name__ == '__main__':
    main()