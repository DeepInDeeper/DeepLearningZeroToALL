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

更多关于`models`的使用教程参考[这里](https://github.com/tensorflow/models/tree/master/tutorials)