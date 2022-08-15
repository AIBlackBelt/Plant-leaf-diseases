# Plant-leaf-diseases

Hello community,

This repo shows work that concerns image classification of different plant leafs species whose health status differs. The dataset used is the one available on kaggle : https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset. it contains different plant image labeled with 38 possible classes. The classes indicate both the plant species that is the first word and its health status second word.

the modeling part has been done in a kaggle notebook here : https://www.kaggle.com/code/majdikarim/alexnet-vgg16-mobilenetv2-inceptionv3-benchmarking
You can download the model in .h5 format from the last notebook cell, it's a model based on inceptionv3.

the script load_model.py can be ran as long as the .h5 file and .py file are hosted locally in the same machine (in an offline way). the script will ask for the path of the image file to process, it should also occur locally. It will load the model use it to make a forward pass on the image after resizing it and produce the plant specie and health status.

![Alt text](/test_images./cap1.png?raw=true) 



