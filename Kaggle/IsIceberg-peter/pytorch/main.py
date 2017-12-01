#coding:utf-8

#sys module
import os,json,time,fire,ipdb,tqdm
import math
import torch


#user module
from utils import opt
import models
from models import diff_net as net
from Iceberg_dataset import readSuffleData,getTrainValLoaders,fixSeed
from iceberg_classfier import generateSingleModel,testModel

batch_size =128
use_cuda = torch.cuda.is_available()
if use_cuda:
    num_workers = 0 # for windows version of PyTorch which does not share GPU tensors
else:
    num_workers = 4


def savePred(df_pred, val_score):
    csv_path = str(val_score) + '_sample_submission.csv'
    df_pred.to_csv(csv_path, columns=('id', 'is_iceberg'), index=None)
    print(csv_path)


def train(**kwargs):
	# train model
	opt.parse(kwargs)
	fixSeed(opt.global_seed)
	for i in range(10):
		print i
		model = models.SimpleNet()
		# model = ResNetLike(BasicBlock, [1, 3, 3, 1], num_channels=2, num_classes=1)
		data, full_img = readSuffleData(seed_num=opt.global_seed)
		train_loader, val_loader, train_ds, val_ds = getTrainValLoaders(data,full_img,batch_size,num_workers)
		# train_loader, val_loader, train_ds, val_ds = getCustomTrainValLoaders()
		model, val_result= generateSingleModel(model,train_loader, val_loader, train_ds, val_ds,num_epoches=55)

		#print (model)
	df_pred = testModel(model)
	savePred(df_pred,val_result)




if __name__ == '__main__':
    fire.Fire()