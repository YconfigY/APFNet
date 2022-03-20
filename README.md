## Attribute-Based Progressive Fusion Network for RGBT Tracking<br>
## This project is created base on<br>
--MDNet: Real-Time Multi-Domain Convolutional Neural Network Tracker Created by Ilchae Jung, Jeany Son, Mooyeol Baek, and Bohyung Han
## Prerequisites<br>
<ol>
  <li> python>=3 </li>	
  <li> pytorch>=1.0 </li>	
  <li> some others library functions </li>	
</ol>
<br>
For more detailed packages, refer to [MDNet](https://github.com/hyeonseobnam/py-MDNet).<br> 

## Pretrained model for APFNet<br>
In our tracker, we use MDNet as our backbone and extend to multi-modal tracker.We use imagenet-vid.pth as our pretrain model.Then we use this with the training model in GTOT and RGBT234 models to pre-train our dual-stream MDNet_RGBT backbone network.And thus we get the **GTOT.pth** and **RGBT234.pth**.And Then We load the basic model to
train Our network and get the final model.Our model and the pretrain model is available at [pth model](https://pan.baidu.com/s/12aR8vmPx7KiHDFkAr7VfwQ).The extract code is **mmic**.<br>

## Run tracker<br>
In the tracking/Run.py file, you need to change dataset path, model_path and result_path In the tracking/Run.py file. You can load the model GTOT_ALL_Transformer for testing RGBT234 and LasHeR. And use the RGBT234_ALL_Transformer for testing the GTOT.<br>

## Train<br>
There Stage train:<br>
At First you should use the GTOT and RGBT234 datasets with challenge tags,you can find the datasets in (https://github.com/mmic-lcl/Datasets-and-benchmark-code) run **prepro_data.py** generate a xxx.pkl file to store the data path.Please note that you should adjust the phased training parameters in the **pretrain_option.py** when training each stage.
<ol>
  <li> In the first stage,you should run the <b>train_stage1.py</b> 5 times because we have five attribute branches. Each time we train the network with specific label data,  we load the pre-trained backbone network model parameters and then add the specific branch one by one for training. Note that at this stage we only save each branch model parameters. </li>	
  <li> You have spawned 5 corresponding challenge branches in one phase.In the second stage,you can load the backbone and all branches parameters for training the Attribute-Based Aggregation Fusion model. you should run the <b>train_stage2.py</b> </li>	
  <li> On the basis of the two-stage, only the model parameters generated by the second stage need to be loaded, and the final model can be generated in the three-stage training. you should run the <b>train_stage3.py</b></li>	
</ol>
