在这个工程，我们将通过使用`TensorFlow`的`models`，在确保这个文本能正确运行，你需要完成这些步骤：
在ubuntu中(TODO:windows 还未测试,可以参考[这篇文章](https://www.urlteam.org/2017/09/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B%E7%AC%94%E8%AE%B0%E4%BA%8C%EF%BC%9Atensorflow%E5%B0%8F%E7%99%BD%E5%AE%9E%E8%B7%B5/)) 


依赖环境：
```
sudo apt-get install protobuf-compiler python-pil python-lxml
sudo pip install jupyter,matplotlib,pillow,lxml
```

依赖`models`,在使用前必须编译Protobuf库.
```
git clone https://github.com/tensorflow/models.git
cd model/object_detection
protoc object_detection/protos/*.proto --python_out=.

```

```
sudo gedit ~/.bashrc #open bashrc file
#then add this new line to the end of your bashrc file
export
PYTHONPATH=$PYTHONPATH=/home/XXX/tensorflow/models/research:/home/XXX/tensorflow/models/research/slim
#then restart then bashrc to make it work(You should change XXX for you own path~)
source ~/.bashrc
#you can change the path to your own!
```

更多关于`models`的使用教程参考[这里](https://github.com/tensorflow/models/tree/master/tutorials)


=====    

更新使用API进行训练自己的数据集：   
使用Tensorflow Object Detection API的关键步骤：   
1. 安装Tensorflow Models   
这里提供了帅小锅的[解决方案][blog-02] 没有尝试使用(超详细的win10安装方案，给个赞)，这里我使用的是[另外一种解决方案][blog-03]   

Mention:
```shell
sudo gedit ~/.bashrc #open bashrc file
#then add this new line to the end of your bashrc file
export
PYTHONPATH=$PYTHONPATH=/home/XXX/tensorflow/models/research:/home/XXX/tensorflow/models/research/slim
#then restart then bashrc to make it work(You should change XXX for you own path~)
source ~/.bashrc
#you can change the path to your own!
```
2. 自己的数据集准备：
这里推荐使用[labelImg][link-01]进行前期标注工作，标注的 `xml`和数据集放在同一个文件夹，即   

images（文件夹）   
  |__ train（文件夹）   
      |__ 01.jpg    
      |__ 01.xml   
  |__ test（文件夹）   
      |__ 01.jpg   
      |__ 01.xml   

3. 数据格式的转换   
对于准备好的数据，需要进行一定格式下的转换。使用到 `xml_to_csv.py` 和 `generate_tfrecord.py`   

- `xml_to_csv.py`  

训练集和测试集 需要更改`12-13行`变更到对应文件夹目录 以及 `40行`保存文件名，这样会在对应文件夹生成 对应的csv文件(如：`train_label.csv`、 `val_label.csv`)。   
- `generate_tfrecord.py` 
`35行`更为为自己的标签
拷贝两个csv文件到 data目录下，变更`92行`的 `test`为对应的训练集 OR 测试集的文件名   
通过键入：  
```shell
python generate_tfrecord.py --csv_input=data/train_label.csv  --output_path=data/train.record
python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=data/val.record

``` 

4. config 文件
在[model-config][link-02]下载config 以及其对应的权重值文件，如  `ssd_mobilenet_v1_coco.config`
将该文件放在`training`文件夹下进行如下的修改：   
- num_classes 更为实际情况；
- bath_size 更改为合适的数值
- `fine_tune_checkpoint: "ssd_mobilenet_v1_coco_11_06_2017/model.ckpt"   
  from_detection_checkpoint: true` 建议删除
- 四处`PATH_TO_BE_CONFIGURED`进行更改，`train` data/train.record;`val` data/val.record (就是你自己保存的tfrecord的路径设置下)，然后这里我们还要一个pbtxt的文件需要设置下。在data目录下新建一个`pbtxt`文件，比如（name.pbtxt）(config文件设置一下，`train`和`val`都设置：data/name.pbtxt )    
```shell
item {  
  id: 1  
  name: 'tv'  
}  
  
item {  
  id: 2  
  name: 'vehicle'  
}  
```   

5.`Tensorflow Models`下`models\research\object_detection`将`train.py`和`export_inference_graph.py` 拷贝到工程目录下。    



查看下整个目录结构：
data（文件夹）
   |__ name.pbtxt   
   |__ train_label.csv   
   |__ val_label.csv   
   |__ train.record    
   |__ val.record   
`ssd_mobilenet_v1_coco_2017_11_17`（文件夹）   
   |__ `frozen_inference_graph.pb`   
images（文件夹）   
  |__ train（文件夹）   
      |__ 01.jpg   
      |__ 01.xml   
  |__ test（文件夹）   
      |__ 01.jpg   
      |__ 01.xml  
training（文件夹）   
  |__ `ssd_mobilenet_v1_coco.config`   
generate_tfrecord.py   
xml_to_csv.py   
train.py       


 6. 训练    
```shell
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_coco.config
```   
训练过程完全可以自行中断，当发现训练差不多的啦（若出现梯度爆炸或者消失，试着减小config里面的学习率的参数），就可以进行转换一下pb格式啦  

在工程目录下新建一个`inference_graph `用来存放graph

```shell
python export_inference_graph.py  --input_type image_tensor  --pipeline_config_path training/ssd_mobilenet_v1_coco.config   --trained_checkpoint_prefix training/model.ckpt-31012   --output_directory inference_graph 

```
（注意修改 `trained_checkpoint_prefix`为自己训练对应的次数），操作无误的话，会在 inference_graph 文件夹下由我们的一些想要的文件。


7. 推断   
可以借鉴 `Tensorflow Models`下`models\research\object_detection`下的`object_detection_tutorial.ipynb`进行修改：
主要修改 ，
```python
#Model preparation  
# What model to download.  
  
#这是我们刚才训练的模型  
MODEL_NAME = 'tv_vehicle_inference_graph'
#对应的Frozen model位置  
# Path to frozen detection graph. This is the actual model that is used for the object detection.  
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.  
PATH_TO_LABELS = os.path.join('training', 'name.pbtxt')
#改成自己例子中的类别数，2  
NUM_CLASSES = 2
```   





非常感谢帅小锅[@Xiang Guo][author-1]提供的视频教程方案以及[CSDN博客][blog-01]

[author-1]:https://github.com/XiangGuo1992
[blog-02]:https://blog.csdn.net/dy_guox/article/details/79081499
[blog-03]:https://www.urlteam.org/2017/09/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B%E7%AC%94%E8%AE%B0%E4%BA%8C%EF%BC%9Atensorflow%E5%B0%8F%E7%99%BD%E5%AE%9E%E8%B7%B5/
[blog-01]:https://blog.csdn.net/dy_guox/article/details/79111949
[link-01]:https://tzutalin.github.io/labelImg/
