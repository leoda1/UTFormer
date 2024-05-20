# UTFormer
 A Lightweight Triple Attention Transformer Network for Semantic Segmentation of UAV Images
---

### 目录
1. [所需环境](#所需环境)
2. [训练步骤](#训练步骤)
3. [预测步骤](#预测步骤)
4. [评估步骤](#评估步骤)

### 所需环境
torch==1.2.0  

### 训练步骤
#### a、训练voc数据集
1、将voc数据集放入dataset中。  
2、在train.py中设置对应参数。  
3、运行train.py进行训练。   

### 预测步骤
#### 使用自己训练的权重
1、按照训练步骤训练。    
2、在pridect.py文件里面，修改model_path、num_classes、backbone使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，num_classes代表要预测的类的数量加1，backbone是所使用的主干特征提取网络**。    
```
3、运行predict.py，输入    
```python
img/yourpth.jpg
```
可完成预测。    
4、在predict.py里面进行设置可以进行fps测试、整个文件夹的测试和video视频检测。   

### 评估步骤 
1、运行get_miou.py即可获得miou大小。  

### Reference
1、local attention的QKV计算 参考https://github.com/qhfan/CloFormer
