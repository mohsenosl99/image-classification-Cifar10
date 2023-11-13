import torch
import torch.nn as nn
from DeepLearning.evaluation import accuracy
import torch
import torch.nn as nn
def train(train_dataset,test_dataset,models,block,batch_size,epochs
          ,learning_rate=1e-3):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    cifar_train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True)

    cifar_val_loader = torch.utils.data.DataLoader(test_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False)


    model = models(block).to(device)



    # Deeple function
    criterion = nn.CrossEntropyLoss()
    # optimzier
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    top1_acc_train = []
    top1_acc_val = []
    loss_avg_train = []
    loss_avg_val = []

    for epoch in range(1, epochs + 1):
        sum_train_acc_epoch = 0
        sum_val_acc_epoch = 0
        sum_train_loss_epoch = 0
        sum_val_loss_epoch = 0
        total=0
    
        model.train()
        mode = "train"
        for batch_idx, (images, labels) in enumerate(cifar_train_loader,1):
            images = images.to(device)
            labels = labels.to(device)
            labels_pred = model(images)
            loss = criterion(labels_pred, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total+=labels.size(0)

            acc1 = accuracy(labels_pred, labels)
            sum_train_acc_epoch += acc1
            sum_train_loss_epoch += loss.detach().item()
            if batch_idx % int(round(len(cifar_train_loader)/6)) ==0:
                print(f"At epoch {epoch}, average accuracy till batch_index --> {batch_idx}: {100.*sum_train_acc_epoch/total}")
        top1_acc_train.append(100.*sum_train_acc_epoch/total)
        loss_avg_train.append(sum_train_loss_epoch / batch_idx)
        
        total=0

        model.eval()
        mode = "val"
        with torch.no_grad():        
            for batch_idx, (images, labels) in enumerate(cifar_val_loader,1):
                images = images.to(device)
                labels = labels.to(device)
                labels_pred = model(images)
                loss = criterion(labels_pred, labels)
                acc1 = accuracy(labels_pred, labels)
                sum_val_acc_epoch += acc1
                sum_val_loss_epoch += loss.detach().item()
                total+=labels.size(0)
            top1_acc_val.append(100.*sum_val_acc_epoch/total)
            loss_avg_val.append(sum_val_loss_epoch / batch_idx)
    return top1_acc_val,top1_acc_train,loss_avg_val,loss_avg_train