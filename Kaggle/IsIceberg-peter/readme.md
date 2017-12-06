## kaggle competion

[Iceberg Classifier Challenge](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/data)

> using satellite data to build a computer vision based surveillance system   




# Data fields

[download here](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/data)

## train.json, test.json

The data \(`train.json`,`test.json`\) is presented in`json`format. The files consist of a list of images, and for each image, you can find the following fields:

* **id**
  * the id of the image
* **band\_1, band\_2**
  * the
    [flattened](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.flatten.html)
    image data. Each band has 75x75 pixel values in the list, so the list has 5625 elements. Note that these values are not the normal non-negative integers in image files since they have physical meanings - these are
    **float**
    numbers with unit being
    [dB](https://en.wikipedia.org/wiki/Decibel)
    . Band 1 and Band 2 are signals characterized by radar backscatter produced from different polarizations at a particular incidence angle. The polarizations correspond to HH \(transmit/receive horizontally\) and HV \(transmit horizontally and receive vertically\). More background on the satellite imagery can be found
    [here](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge#Background)
    .
* **inc\_angle**
  * the incidence angle of which the image was taken. Note that this field has missing data marked as "na", and those images with "na" incidence angles are all in the training data to prevent leakage.
* **is\_iceberg**
  * the target variable, set to 1 if it is an iceberg, and 0 if it is a ship. This field only exists in
    `train.json`
    .

![图片](/img/kaggle_IsIceberg.PNG)
