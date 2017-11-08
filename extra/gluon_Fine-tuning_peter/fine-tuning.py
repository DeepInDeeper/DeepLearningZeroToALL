# -*- coding: utf-8 -*- 

from mxnet import gluon
from mxnet import nd
from mxnet import image
from mxnet import init
import sys
import utils
import zipfile


data_dir = '/media/yijie/娱乐/tmp/hotdog'
'''
fname = gluon.utils.download(
    'https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/hotdog.zip',
    path=data_dir, sha1_hash='fba480ffa8aa7e0febbb511d181409f899b9baa5')
'''
fname = '/media/yijie/娱乐/tmp/hotdog/hotdog.zip'
with zipfile.ZipFile(fname, 'r') as f:
    f.extractall(data_dir)

## pic augs
train_augs = [
    image.HorizontalFlipAug(.5),
    image.RandomCropAug((224,224))
]

test_augs = [
    image.CenterCropAug((224,224))
]

def transform(data, label, augs):
    data = data.astype('float32')
    for aug in augs:
        data = aug(data)
    data = nd.transpose(data, (2,0,1))
    return data, nd.array([label]).asscalar().astype('float32')

train_imgs = gluon.data.vision.ImageFolderDataset(
    data_dir+'/hotdog/train',                                   
    transform=lambda X, y: transform(X, y, train_augs))
test_imgs = gluon.data.vision.ImageFolderDataset(
    data_dir+'/hotdog/test',                                   
    transform=lambda X, y: transform(X, y, test_augs))


# get resnet18 version2 from Internet 
# path: /home/yijie/.mxnet/models
from mxnet.gluon.model_zoo import vision as models
pretrained_net = models.resnet18_v2(pretrained=True)
pretrained_net.classifier
pretrained_net.features[1].params.get('weight').data()[0][0]

finetune_net = models.resnet18_v2(classes=2)
finetune_net.features = pretrained_net.features
finetune_net.classifier.initialize(init.Xavier())

sys.path.append('..')
def train(net, ctx, batch_size=64, epochs=10, learning_rate=0.01, wd=0.001):
    train_data = gluon.data.DataLoader(train_imgs, batch_size, shuffle=True)
    test_data = gluon.data.DataLoader(test_imgs, batch_size)

    # 确保net的初始化在ctx上
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    # 训练
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {
        'learning_rate': learning_rate, 'wd': wd})
    utils.train(train_data, test_data, net, loss, trainer, ctx, epochs)

# fine-tuning
ctx = utils.try_all_gpus()
print("the following result is from fine-tuning......\n")
train(finetune_net, ctx)


# random init
scratch_net = models.resnet18_v2(classes=2)
scratch_net.initialize(init=init.Xavier())
print("the following result is from random-init.....\n")
train(scratch_net, ctx)


# do some predictions
def classify_hotdog(net, fname):
    with open(fname, 'rb') as f:
        img = image.imdecode(f.read())        
    data, _ = transform(img, -1, test_augs)
    plt.imshow(data.transpose((1,2,0)).asnumpy()/255)
    data = data.expand_dims(axis=0)
    out = net(data.as_in_context(ctx[0]))
    out = nd.SoftmaxActivation(out)
    pred = int(nd.argmax(out, axis=1).asscalar())
    prob = out[0][pred].asscalar()
    label = train_imgs.synsets
    return 'With prob=%f, %s'%(prob, label[pred])

classify_hotdog(finetune_net, '../img/real_hotdog.jpg')
classify_hotdog(finetune_net, '../img/leg_hotdog.jpg')
classify_hotdog(finetune_net, '../img/dog_hotdog.jpg')
