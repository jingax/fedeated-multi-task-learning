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
out_epochs = 50  ## equvalent to epochs (no differnce)
batch_size = 2
lr = [0.01]*n_clients    # try different learning rates for different clients
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
    if (exists(f'no_colab_model_{i}.pt')):
        checkpoint = torch.load(f'no_colab_model_{i}.pt')
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

if (exists(f'no_colab_dummy.npy')):
    acc = np.load('no_colab_dummy.npy')
    for i in range(n_clients):
        test_accuracies[i] = list(acc[i])[0:epoch_done+1]



## Can change?
global_criterion = nn.MSELoss()


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
            outputs = client_model[client_id](inp)
            labels = torch.reshape(labels, outputs.shape)
            labels = labels.to(device)
            loss = global_criterion(outputs, labels)
            loss.backward()
            optimizers[client_id].step()
            running_loss += loss.item()
            # print statistics
            	

            if i % 100 == 99:    # print every 1000 mini-batches
                print(f'    ==>[epoch :{out_epoch + 1}, client :{client_id}, batch processed: {i + 1:5d}] train loss: {running_loss} ')
                running_loss = 0.0
            i += 1

        #######################################################
        ##################### Test models #####################
        client_model[client_id].eval()  # handle drop-out/batch norm layers
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
            }, f"no_colab_model_{client_id}.pt")
        test_accuracies_dummy = np.array(test_accuracies)

        np.save("no_colab_dummy.npy",test_accuracies_dummy)

for i in range(n_clients):
    print(test_accuracies[i])

test_accuracies = np.array(test_accuracies)

np.save("no_colab.npy",test_accuracies)
