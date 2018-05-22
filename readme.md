# Efficient Deep Learning for Stereo Matching -- A Tensorflow implementation

This is a tensorflow implementation of the paper **"Efficient Deep Learning for Stereo Matching" by by Luo, W. and Schwing, A. and Urtasun, R**. For more details, please refer the original torch implementation of the authors at [https://bitbucket.org/saakuraa/cvpr16_stereo_public](https://bitbucket.org/saakuraa/cvpr16_stereo_public) 

A key feature of this implementation is the use of (recent) tensorflow Dataset API -- the input pipeline for the deep network. With this, the model gets trained for 40K iterations in less than 4 hours. This significantly quicker than the original torch based implementation of this paper. Note the same GPU is used in both cases.

The trained model on the almost full validation set (2560000 samples) of kitti2012 produces a **3-pixel error of 7.23%**. 

Following are the **loss, training and validation accuracy curves on kitti2012 dataset** for 40K iterations. 

The step on x-axis is EPOCH count. 1 EPOCH equals 100 iterations. The batch size is 128. Training accuracy is averaged over an EPOCH and validation accuracy is counted on 12800 samples after every EPOCH. 

![Loss Curve for kitti2012 dataset (160 training images)](/images/train_loss.png)


![Training Curve for kitti2012 dataset](/images/train_acc.png)


![Validation curve for kitti2012 dataset (34 validation images)](/images/val_acc.png)

*To train a model with default settings, please set train_mode param to True in train_model_slim.py file* 

*To validate a trained model, please set train_mode param to False in train_model_slim.py file*
