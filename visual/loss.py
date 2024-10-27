import torch
import os.path
from cProfile import label

import matplotlib.pyplot as plt
from fontTools.unicodedata import block
from timm.data.auto_augment import color

class LossVisula():
    def __init__(self, loss_list, model_title, dataset_name, n_scale, traintime, epochs, batch_size, epoch, form='single'):
        self.loss_list = loss_list
        self.dir_path = './loss_picture/'+dataset_name+'_'+str(n_scale)
        self.name = model_title+"_allepochs="+str(epochs)+"_batchsize="+str(batch_size)+"_"+traintime+"_epoch="+str(epoch)+"_"+form+'.png'
        self.save_path =self.dir_path + '/' +self.name
        self.form = form

    def plot_loss(self):

        save_path = os.path.dirname(self.save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if self.form == 'single':
            loss_list_cpu = [loss.cpu().item() for loss in self.loss_list]
        else:
            loss_list_cpu = self.loss_list

        plt.figure(figsize=(10, 5))
        plt.plot(loss_list_cpu, label='Loss', color='blue', marker='o')
        plt.title('Loss Visualization')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.savefig(self.save_path)
        plt.show(block = False)
        plt.close()
        #plt.show()