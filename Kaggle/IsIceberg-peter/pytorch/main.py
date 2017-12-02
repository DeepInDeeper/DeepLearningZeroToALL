#coding:utf-8

#sys module
import os,json,time,fire,ipdb,tqdm
import math
import torch


#user module
from utils import opt,Visualizer
import models
from models import diff_net as net
from Iceberg_dataset import readSuffleData,getTrainValLoaders,getCustomTrainValLoaders,fixSeed
from iceberg_classfier import generateSingleModel,testModel


use_cuda = torch.cuda.is_available()
if use_cuda:
    num_workers = 0
else:
    num_workers = 4



vis = Visualizer(env=opt.env)


def savePred(df_pred, val_score):
    csv_path = str(val_score) + '_sample_submission.csv'
    df_pred.to_csv(csv_path, columns=('id', 'is_iceberg'), index=None)
    print(csv_path)


def train(**kwargs):
	# train model
	opt.parse(kwargs)
	batch_size,validationRatio = opt.batch_size,opt.validationRatio
	vis.vis.env = opt.env
	LR = opt.LR
	fixSeed(opt.global_seed)
	for i in range(60):
		print ("Ensamble number:" + str(i))
		model = models.SimpleNet()
		#model = models.ResNetLike(BasicBlock, [1, 3, 3, 1], num_channels=2, num_classes=1)
		data, full_img = readSuffleData(opt.global_seed,opt.BASE_FOLDER)
		train_loader, val_loader, train_ds, val_ds = getTrainValLoaders(data,full_img,batch_size,num_workers,validationRatio)
		#train_loader, val_loader, train_ds, val_ds = getCustomTrainValLoaders(data,full_img,batch_size,num_workers)			
		if epoch > lr_period and epoch % 10 == 0:
			LR = LR * 0.5
		model, val_result,train_result= generateSingleModel(model,train_loader, val_loader, train_ds, val_ds,LR,opt.global_epoches)	
		vis.plot('val_loss', val_result)
		vis.plot("train_loss",train_result)
		#print (model)
	df_pred = testModel(model)
	savePred(df_pred,val_result)




if __name__ == '__main__':
    fire.Fire()