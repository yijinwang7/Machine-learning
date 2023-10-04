import torch
import torchvision
import torch.nn as nn
import numpy as np
import json
import utils
import validate
import argparse
import models.densenet
import time
import dataloaders.datasetaug
import dataloaders.datasetnormal

params = utils.Params("..\\Assignment4\\config\\esc_densenet.json")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_loader = dataloaders.datasetaug.fetch_dataloader("{}testing128mel{}.pkl".format(params.data_dir, 1),
                                                     params.dataset_name, params.batch_size, params.num_workers,
                                                     'validation')

model_opt = models.densenet.DenseNet(params.dataset_name, params.pretrained).to(device)
string1 = "D:\\Dataset\\ESC\\ESC-sub\\checkpoint\\model_optimal.tar"
utils.load_checkpoint(string1, model_opt)
acc= validate.evaluate(model_opt, device, test_loader)

print(acc)


model_nWD = models.densenet.DenseNet(params.dataset_name, params.pretrained).to(device)
string2 = "D:\\Dataset\\ESC\\ESC-sub\\checkpoint\\model_noWD.tar"
utils.load_checkpoint(string2, model_nWD)
acc2= validate.evaluate(model_nWD, device, test_loader)
print(acc2)

test_loader2 = dataloaders.datasetnormal.fetch_dataloader("{}testing128mel{}.pkl".format(params.data_dir, 1),
                                                     params.dataset_name, params.batch_size, params.num_workers)
model_norm = models.densenet.DenseNet(params.dataset_name, params.pretrained).to(device)
string3 = "D:\\Dataset\\ESC\\ESC-sub\\checkpoint\\model_norm.tar"
utils.load_checkpoint(string2, model_norm)
acc3= validate.evaluate(model_norm, device, test_loader2)
print(acc3)


model_rand = models.densenet.DenseNet(params.dataset_name, params.pretrained).to(device)
string4 = "D:\\Dataset\\ESC\\ESC-sub\\checkpoint\\model_rand.tar"
utils.load_checkpoint(string3, model_rand)
acc2= validate.evaluate(model_rand, device, test_loader)
print(acc2)


