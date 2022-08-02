import numpy as np
import torch
import torch.nn as nn
import models as mdl
import torchvision
from PIL import Image
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import keras
from keras.preprocessing import image
import tensorflow as tf
from torch.utils.data import Dataset, DataLoader, TensorDataset
from os.path import exists
import models as mdl


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Current cuda device: ',torch.cuda.get_device_name(0))



##############################################
##############################################
n_clients = 3
data_length = 20000
out_epochs = 50       ## equvalent to epochs (no differnce)
batch_size = 2
lr = [0.01]*n_clients    # try different learnign rates for different clients
lambda_kd = 0.1
sample_size = 5
##############################################
##############################################



df = pd.read_csv('./dataset/anno_csv.csv')
train_image = []
print("Loading Images...")

for i in tqdm(range(data_length)):
    img = keras.utils.load_img('./dataset/img_align_celeba/'+df['file_name'][i],target_size=(178,218,3))
    img = keras.utils.img_to_array(img)
    img = img/255
    train_image.append(img.T)
X = np.array(train_image)

df = df.head(n=data_length)
y = np.array(df.drop(['q', 'file_name'],axis=1))
y = y.astype('float32')

# X = torch.from_numpy(X)
#####################
# Self Note : - Use forch.from_numpy for conversion nahi toh error aayega 



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)

del df
del X
del y


##############################################
########## Split data into n clients #########

permute_train = list(np.random.permutation(y_train.shape[0]))
permute_test = list(np.random.permutation(y_test.shape[0]))
client_data = []
client_model = []
client_train_data_size = int(y_train.shape[0]/n_clients)
client_test_data_size = int(y_test.shape[0]/n_clients)
test_accuracies = []

## Maybe try with different optimizers for different clients
optimizers = []
epoch_done = -1




for i in tqdm(range(n_clients)):
    test_accuracies.append([])
    # ingore first value (for by passing pytohns optimizations)
    if (exists(f'colab_model_{i}.pt')):
        checkpoint = torch.load(f'colab_model_{i}.pt')
        model = mdl.ResNet9(in_channels=3, feat_dim=100, output_shape=1)
        model.load_state_dict(checkpoint['model_state_dict'])
        client_model.append(model)
        epoch_done = epoch = checkpoint['epoch']
        client_model[i].to(device)
        optimizers.append(torch.optim.SGD(client_model[i].parameters(), lr=lr[i]))
        optimizers[i].load_state_dict(checkpoint['optimizer_state_dict'])

        
    else:
        client_model.append(mdl.ResNet9(in_channels=3, feat_dim=100, output_shape=1))
        client_model[i].to(device)
        optimizers.append(torch.optim.SGD(client_model[i].parameters(), lr=lr[i]))
    
    client_data.append({
        'X_train': torch.from_numpy(X_train[permute_train[i*client_train_data_size:(i+1)*client_train_data_size]]),
        'X_test' : torch.from_numpy(X_test[permute_test[i*client_test_data_size:(i+1)*client_test_data_size]]),
        'y_train': torch.from_numpy(y_train[permute_train[i*client_train_data_size:(i+1)*client_train_data_size]]),
        'y_test' : torch.from_numpy(y_test[permute_test[i*client_test_data_size:(i+1)*client_test_data_size]])
    })

del X_train
del X_test
del y_train
del y_test

if (exists(f'yes_colab_dummy.npy')):
    acc = np.load('yes_colab_dummy.npy')
    for i in range(n_clients):
        test_accuracies[i] = list(acc[i])[0:epoch_done+1]

## Can change?
global_criterion = nn.MSELoss()
kd_criterion = nn.MSELoss()


##############################################################################################
######### Initialize class buffers .......... ..

class_buffer = []
for client_id in range(n_clients):
    # 2 is the number of classes per task (binary classification)
    class_buffer.append([])
    for cls_ in range(2): 
        idx = (client_data[client_id]["y_train"][:,client_id]==cls_).nonzero().squeeze()
        idx = idx[torch.randperm(len(idx))[:sample_size]]
        # print(client_data[client_id]["y_train"][:,client_id][idx])
        sampled_x = client_data[client_id]["X_train"][idx]
        client_model[client_id].eval()
        sampled_x = sampled_x.to(device)
        features = client_model[client_id].features(sampled_x)
        features = torch.sum(features,dim=0)/sampled_x.shape[0]
        # print(features.shape)
        distribution = {}
        for task_id in range(n_clients):
            temp_ = client_data[client_id]["y_train"][idx][:,task_id]
            tot_points = temp_.shape[0]
            tot_positives = torch.numel(temp_[temp_>0.5])
            # print(tot_points," ",tot_positives)
            distribution[task_id] = [tot_points-tot_positives,tot_positives]
        class_buffer[client_id].append({'features':features.detach(),'distribution':distribution})



##############################################################################################
######### Train the models dun dun dun dun dun...... ..


for out_epoch in range(epoch_done+1,out_epochs):
    for client_id in range(n_clients):
        client_model[client_id].train()
        ds = TensorDataset(client_data[client_id]["X_train"], client_data[client_id]["y_train"][:,client_id])
        dl = DataLoader(ds, batch_size=batch_size)
        running_loss = 0.0
        i = 0
        for inp, labels in dl:
            optimizers[client_id].zero_grad()
            inp = inp.to(device)
            features = client_model[client_id].features(inp) 
            outputs = client_model[client_id].classifier(features)
            labels = torch.reshape(labels, outputs.shape)
            labels = labels.to(device)
            loss = global_criterion(outputs, labels)
            
            ##########################################################
            ############ KD loss based on distribution ###############
            ##########################################################
            
            for task_id in range(n_clients):
                if task_id == client_id:
                    continue
                for cls_ in range(2): 
                    teacher_features = class_buffer[task_id][cls_]["features"]
                    tot_points = class_buffer[task_id][cls_]["distribution"][client_id][0] + class_buffer[client_id][cls_]["distribution"][client_id][1]
                    if tot_points == 0:
                        continue
                    teacher = teacher_features.clone().detach()
                    #for i in range(1,batch_size):
                    #    teacher = torch.cat((teacher,teacher_features),1)
                    #print(teacher.shape)
                    #print(features.shape)
                    for batch_no in range(batch_size):
                        loss += lambda_kd * class_buffer[task_id][cls_]["distribution"][client_id][int(labels[batch_no])] * kd_criterion(features[batch_no],teacher)/tot_points

            ##########################################################
            
            
            loss.backward()
            optimizers[client_id].step()
            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                print(f'    ==>[epoch :{out_epoch + 1}, client :{client_id}, batch processed: {i + 1:5d}] train loss: {running_loss}')
                running_loss = 0.0
            i += 1

        #######################################################
        ##################### Test models #####################
        #######################################################
        
        client_model[client_id].eval() 
        test_loss = 0
        ds_test = TensorDataset(client_data[client_id]["X_test"], client_data[client_id]["y_test"][:,client_id])
        dl_test = DataLoader(ds, batch_size=batch_size)
        
        with torch.no_grad():
            for x,y in dl_test:
                x = x.to(device)
                out = client_model[client_id](x)
                y = torch.reshape(y, out.shape)
                y = y.to(device)
                test_loss += global_criterion(out, y)
        val_loss = test_loss / len(dl_test)
        print(f"[epoch :{out_epoch + 1}, client :{client_id}] Test loss: {test_loss}")
        test_accuracies[client_id].append(test_loss.item())

        ############################################################
        ########### Checkpoint after every XXXXXXXXXXXXXX ##########
        ############################################################

        if(out_epoch%1==0):
            torch.save({
            'epoch': out_epoch,
            'model_state_dict': client_model[client_id].state_dict(),
            'optimizer_state_dict': optimizers[client_id].state_dict(),
            }, f"colab_model_{client_id}.pt")
        
        
        ############################################################
        ########### Update represenatations at each epoch ##########
        ############################################################

        for cls_ in range(2): 
            idx = (client_data[client_id]["y_train"][:,client_id]==cls_).nonzero().squeeze()
            idx = idx[torch.randperm(len(idx))[:sample_size]]
            # print(client_data[client_id]["y_train"][:,client_id][idx])
            sampled_x = client_data[client_id]["X_train"][idx]
            client_model[client_id].eval()
            sampled_x = sampled_x.to(device)
            features = client_model[client_id].features(sampled_x)
            features = torch.sum(features,dim=0)/sampled_x.shape[0]
            # print(features.shape)
            distribution = {}
            for task_id in range(n_clients):
                temp_ = client_data[client_id]["y_train"][idx][:,task_id]
                tot_points = temp_.shape[0]
                tot_positives = torch.numel(temp_[temp_>0.5])
                # print(tot_points," ",tot_positives)
                distribution[task_id] = [tot_points-tot_positives,tot_positives]
            class_buffer[client_id][cls_]['features'] = features.detach()
            class_buffer[client_id][cls_]['distribution'] = distribution
            

            ###################################################
            ############# Representations updated #############
            ###################################################
    test_accuracies_dummy = np.array(test_accuracies)

    np.save("yes_colab_dummy.npy",test_accuracies_dummy)



for i in range(n_clients):
    print(test_accuracies[i])

test_accuracies = np.array(test_accuracies)

np.save("yes_colab.npy",test_accuracies)
