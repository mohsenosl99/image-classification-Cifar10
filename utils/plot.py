import matplotlib.pyplot as plt
def plot(top1_acc_val,top1_acc_train,loss_avg_val,loss_avg_train,block):
    fig,ax=plt.subplots(2,1,figsize=(10,15))
    ax[0].plot(top1_acc_train,label='trian')
    ax[0].plot(top1_acc_val,label='validation')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_title(f'Model Accuracy for {block}')
    ax[1].plot(loss_avg_train,label='trian')
    ax[1].plot(loss_avg_val,label='validation')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].set_title(f'Model Loss for {block}')

    ax[0].legend()
    ax[1].legend()
    plt.show()
