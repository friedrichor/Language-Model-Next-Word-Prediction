# Language-Model-Next-Word-Prediction
Next word prediction using language models

讲解：[语言模型 实现 下一单词预测（next-word prediction）](https://blog.csdn.net/Friedrichor/article/details/126074843)


数据集说明:
  
<img src="https://user-images.githubusercontent.com/70964199/178480906-ee851fb3-c05a-4422-a536-cf0490b4e335.png" width="50%">

更换模型：  
1. 只需在train.py中更改model = TextRNN(n_class).to(device)即可。
2. 代码中提供的模型为NNLM、RNNLM、RNNLM based on Attention，在model.py文件中。  
当然也可以自己编写其他语言模型，直接调用即可，更改方式同上。
