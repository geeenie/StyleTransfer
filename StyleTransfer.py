#!/usr/bin/env python
# coding: utf-8

# 0. Package Installation

# In[1]:


import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import time
import re


# In[2]:


from utils.imageloader import load_images
from utils.save_image import save_image
from utils.dataloader import load_data
from utils.normalize import batch_normalize
from utils.gram_matrix import gram_matrix
from model.VGG16 import VGG16
from model.TransformerNet import TransformerNet


# 1. Hyperparameter Setting

# In[3]:


batch_size = 8
num_epoch = 10
learning_rate = 1e-4
content_weight = 1e5
style_weight = 1e10
log_interval = 50
ckpt_dir = './checkpoints'
season1 = 'winter'
season2 = 'spring'
method = 'shake_data' # 'one_season', 'shake_data', 'shake_feature'


# 2. Style Images and Train Data Loading

# In[4]:


if(method == 'one_season' and season1 == season2):
    season = season1
    style_data = load_images('./data/', season)
elif(method == 'shake_data' and season1 != season2):
    season = season1 + '_' + season2 + '_data'
    style_data1 = load_images('./data/', season1)
    style_data2 = load_images('./data/', season2)
    idx1 = np.random.choice(10, 5)
    idx2 = np.random.choice(10, 5)
    style_data = torch.stack([style_data1[i] for i in idx1] + [style_data2[i] for i in idx2],0)
elif(method == 'shake_feature' and season1 != season2):
    season = season1 + '_' + season2 + '_feature'
    style_data1 = load_images('./data/', season1)
    style_data2 = load_images('./data/', season2)
    style_data = style_data1
else:
    season = season1
    style_data = load_images('./data/', season)
print(style_data.shape)

train_dataset, train_dataloader, val_dataset, val_dataloader = load_data('./data/', batch_size)
print(train_dataset[0][0].shape)
print(len(train_dataset), len(val_dataset))


# 3. Style Transform with gram

# In[5]:


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


# In[6]:


transformer = TransformerNet()
vgg = VGG16(requires_grad=False).to(device)


# In[7]:


if(method == 'one_season' and season1 == season2):
    features_style = vgg(batch_normalize(style_data.to(device)))
    gram_style = [gram_matrix(y) for y in features_style]
elif(method == 'shake_data' and season1 != season2):
    features_style = vgg(batch_normalize(style_data.to(device)))
    gram_style = [gram_matrix(y) for y in features_style]
elif(method == 'shake_feature' and season1 != season2):
    features_style1 = vgg(batch_normalize(style_data1.to(device)))
    features_style2 = vgg(batch_normalize(style_data2.to(device)))
    gram_style = [gram_matrix((y1+y2)/2) for y1, y2 in zip(features_style1, features_style2)]
else:
    features_style = vgg(batch_normalize(style_data.to(device)))
    gram_style = [gram_matrix(y) for y in features_style]


# 4. TransformerNet training with train data

# In[8]:


optimizer = optim.Adam(transformer.parameters(), lr=learning_rate)
loss_function = nn.MSELoss()


# In[9]:


def train():
    for epoch in range(num_epoch):
        transformer.to(device)
        transformer.train()
        train_content_loss = 0.
        train_style_loss = 0.
        count = 0

        for batch_id, (x, _) in enumerate(train_dataloader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()

            x = x.to(device)
            y = transformer(x)

            y = batch_normalize(y)
            x = batch_normalize(x)

            features_y = vgg(y)
            features_x = vgg(x)

            content_loss = content_weight * loss_function(features_y.relu2_2, features_x.relu2_2)

            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = gram_matrix(ft_y)
                style_loss += loss_function(gm_y, gm_s[:n_batch, :, :])
            style_loss *= style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            train_content_loss += content_loss.item()
            train_style_loss += style_loss.item()

            if (batch_id + 1) % log_interval == 0:
                val_content_loss = 0.
                val_style_loss = 0.
                
                for (x, _) in val_dataloader:
                    n_batch = len(x)
                    x = x.to(device)
                    y = transformer(x)

                    y = batch_normalize(y)
                    x = batch_normalize(x)

                    features_y = vgg(y)
                    features_x = vgg(x)
                    
                    content_loss = content_weight * loss_function(features_y.relu2_2, features_x.relu2_2)

                    style_loss = 0.
                    for ft_y, gm_s in zip(features_y, gram_style):
                        gm_y = gram_matrix(ft_y)
                        style_loss += loss_function(gm_y, gm_s[:n_batch, :, :])
                    style_loss *= style_weight
                    
                    val_content_loss += content_loss.item()
                    val_style_loss += style_loss.item()
                
                msg = "{}\tEpoch {}:[{}/{}]\ttrain\t[content: {:.4f}\tstyle: {:.4f}\ttotal: {:.4f}]".format(
                    time.ctime(), epoch + 1, count, len(train_dataset),
                    train_content_loss / (batch_id + 1),
                    train_style_loss / (batch_id + 1),
                    (train_content_loss + train_style_loss) / (batch_id + 1)
                )
                print(msg)
                print(epoch + 1)
                msg = "\t\t\t\t\t\t\tval\t[content: {:.4f}\tstyle: {:.4f}\ttotal: {:.4f}]".format(
                    val_content_loss / len(val_dataset),
                    val_style_loss / len(val_dataset),
                    (val_content_loss + val_style_loss) / len(val_dataset)
                )
                print(msg)
                
        # 4.1 Save Model
        transformer.eval().cpu()
        ckpt_model_filename = "ckpt_epoch_" + str(epoch + 1) + "_" + season + ".pth"
        print(str(epoch + 1), "th checkpoint is saved!")
        ckpt_model_path = os.path.join(ckpt_dir, ckpt_model_filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': transformer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss
        }, ckpt_model_path)

        transformer.to(device).train()


# In[10]:

print("trainstart")
train()


# 5. Test

# In[11]:


def test():
    content_data = load_images('./data/', 'test')
    content_data = content_data.to(device)
    with torch.no_grad():
        style_model = TransformerNet()
        
        ckpt_model_path = os.path.join(ckpt_dir, f"ckpt_epoch_{num_epoch}_{season}.pth") #FIXME
        checkpoint = torch.load(ckpt_model_path, map_location=device)
        
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(checkpoint.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del checkpoint[k]
        
        style_model.load_state_dict(checkpoint['model_state_dict'])
        style_model.to(device)
        
        output = style_model(content_data).cpu()
        save_image(f'./output_{season}.png', output[0])


# In[12]:

print("teststart")
test()

