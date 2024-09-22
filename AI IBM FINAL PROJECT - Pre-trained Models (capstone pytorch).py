
#AI IBM FINAL PROJECT
#Pre-trained-Models with PyTorch

# These are the libraries will be used for this lab.
import torchvision.models as models
from PIL import Image
import pandas
from torchvision import transforms
import torch.nn as nn
import time
import torch 
import matplotlib.pylab as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py
import os
import glob
torch.manual_seed(0)

from matplotlib.pyplot import imshow
import matplotlib.pylab as plt
from PIL import Image
import pandas as pd
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("the device type is", device)


# Create your own dataset object

class Dataset(Dataset):

    # Constructor
    def __init__(self,transform=None,train=True):
        directory="./concrete_data_week3/"
        train_positive="train/positive/"
        train_negative='train/negative/'
        valid_positive = 'valid/positive/'
        valid_negative = 'valid/negative/'

        positive_file_path=os.path.join(directory,train_positive if train else valid_positive)
        negative_file_path=os.path.join(directory,train_negative if train else valid_negative)
        positive_files=[os.path.join(positive_file_path,file) for file in os.listdir(positive_file_path) if file.endswith(".jpg")]
        negative_files=[os.path.join(negative_file_path,file) for file in os.listdir(negative_file_path) if file.endswith(".jpg")]
        number_of_samples=len(positive_files)+len(negative_files)

        self.all_files=[None]*number_of_samples
        self.all_files[::2]=positive_files
        self.all_files[1::2]=negative_files 
        # The transform is goint to be used on image
        self.transform = transform
        #torch.LongTensor
        self.Y=torch.zeros([number_of_samples]).type(torch.LongTensor)
        self.Y[::2]=1
        self.Y[1::2]=0
        self.len = len(self.all_files)  

    
    # Get the length
    def __len__(self):
        return self.len
    
    # Getter
    def __getitem__(self, idx):            
        image=Image.open(self.all_files[idx])
        y=self.Y[idx]
                  
        # If there is any transform method, apply it onto the image
        if self.transform:
            image = self.transform(image)

        return image, y
    
print("done")

# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]
# train_dataset[0][0].std([1,2])

composed = transforms.Compose([transforms.ToTensor(),
                               transforms.Resize((224, 224)),
                               #transforms.Normalize(mean, std)
                              ])

train_dataset = Dataset(train=True, transform=composed)
validation_dataset = Dataset(train=False, transform=composed)
print("done")

print('Train dataset size: ', train_dataset.len)
print('Valid dataset size: ', validation_dataset.len)


#QUESTION 1 =>
# Step 1: Load the pre-trained model resnet18
model = models.resnet18(pretrained=True)

# Step 2: Set the parameter cannot be trained for the pre-trained model
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(512,2)
model.to(device)

#QUESTION 2 =>
# Step 1: Create the loss function
criterion = nn.CrossEntropyLoss()
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=100)
optimizer = torch.optim.Adam([parameters  for parameters in model.parameters() if parameters.requires_grad],lr=0.001)

n_epochs=1
loss_list=[]
accuracy_list=[]
correct=0
N_test=len(validation_dataset)
N_train=len(train_dataset)
start_time = time.time()

Loss=0
start_time = time.time()
for epoch in range(n_epochs):
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        model.train() 
        # clear gradient
        optimizer.zero_grad()
        # make a prediction 
        z = model(x)
        # calculate loss 
        loss = criterion(z, y)
        # add loss to list
        loss_list.append(loss.item())
        # calculate gradients of parameters 
        loss.backward()
        # update parameters 
        optimizer.step()

    correct=0
    for x_test, y_test in validation_loader:
        x_test, y_test=x_test.to(device), y_test.to(device)
        # set model to eval 
        model.eval()
        #make a prediction 
        z = model(x_test)
        #find max 
        _, yhat = torch.max(z.data, 1)
        #Calculate misclassified  samples in mini-batch 
        correct +=(yhat==y_test).sum().item()
           
    accuracy=correct/N_test

current_time = time.time()
elapsed_time = current_time - start_time
print("elapsed time", elapsed_time, 'Accuracy: ',accuracy)

accuracy
plt.plot(loss_list)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.show()

#QUESTION 3 =>
def show_data(data_sample):
    plt.imshow(data_sample[0])
    plt.title('y = ' + str(data_sample[1]))

count = 0
for x, y in validation_dataset:
    x = torch.unsqueeze(x, 0)
    x, y = x.to(device), y.to(device)
    model.eval()
    z = model(x)
    _, yhat = torch.max(z.data, 1)
    if yhat != y:
        print('Real: ',y.item(),' - Predicted: ',yhat.item() )
        plt.imshow(transforms.ToPILImage()(x[0]), interpolation="bicubic")
        plt.show()
        count += 1
    if count >= 5:
        break 




































