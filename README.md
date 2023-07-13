# 当代人工智能实验五
多模态情感分析

给定配对的文本和图像，预测对应的情感标签
三分类任务：positive, neutral, negative

## module
本次实验所需module如下
- chardet==3.0.4
- numpy==1.22.4+mkl
- pandas==1.3.4
- Pillow==10.0.0
- scikit_learn==1.3.0
- torchvision==0.15.1+cu118
- tqdm==4.62.3
- transformers==4.27.1
- torch==2.0.0+cu118


安装
```shell
pip install -r requirements.txt
```

## 文件结构
```
|-- dataset  #保存了实验数据
|-- image  #报告中图片
|-- models  #模型代码
|-- requirements.txt
|-- README.md
|-- config.py
|-- main.py
|-- data_pro.py
|-- output  #预测结果
|-- utils.py
```

## 训练和指标
- 训练
```shell
python main.py --output_dir output/ --train_filepath dataset/train.json --dev_filepath dataset/dev.json --test_filepath dataset/test.json --do_train
```

- 测试
```shell
python main.py --output_dir output/ --train_filepath dataset/train.json --dev_filepath dataset/dev.json --test_filepath dataset/test.json --do_test --store_preds
```

## 参数选择
```shell
--output_dir  预测文件位置
--train_filepath  训练集位置
--dev_filepath  验证集位置
--test_filepath  测试集位置
--do_train  是否训练模式
--do_test  是否测试模式
--store_preds  是否保存预测结果
...
```

## 实验结果
text_only_loss：0.97
image_only_loss：0.90
multi_loss：0.80
accuracy:0.69