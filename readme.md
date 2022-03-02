# PetFinder.my - Pawpularity Contest

![avatar](https://storage.googleapis.com/kaggle-competitions/kaggle/25383/logos/header.png?t=2021-08-31-18-49-29)

Predict the popularity of shelter pet photos

https://www.kaggle.com/c/petfinder-pawpularity-score

## 比赛介绍

本次竞赛中，参赛者将分析**原始图片和元数据**，预测宠物照 片的 **"受欢迎程度"：Pawpularity**。

- 竞赛类型：本次竞赛属于**计算机视觉-图像分类**，所以推荐使用的模型：Swin-Transformer
- 赛题数据：赛题数据图像的数量适中，但和图像和元数据噪音较大，和标签Pawpularity之间关联性较小，不太好训练。 训练集：9912张图片，隐藏的测试集：大约是6800张图片。
- 评估标准：**RMSE (Root Mean Squared Error)** 。
- 推荐阅读 Kaggle 内的一篇 EDA（探索数据分析）来获取一些预备知识：[Petfinder Pawpularity EDA & fastai starter 🐱🐶 | Kaggle](https://www.kaggle.com/tanlikesmath/petfinder-pawpularity-eda-fastai-starter)

## 数据说明

官方数据页面 [PetFinder.my - Pawpularity Contest | Kaggle](https://www.kaggle.com/c/petfinder-pawpularity-score/data)

赛题数据图像的数量适中，但和图像和元数据噪音较大，和标签Pawpularity之间关联性较小，不太好训练。 训练集：9912张图片，隐藏的测试集：大约是6800张图片。

关于元数据：

主办方提供了所有照片的元数据，其内容为每张照片的关键视觉质量和构图参数进行了手动标注。

•**焦点** - 宠物在不杂乱的背景下很突出，不要太近/太远。

•**眼睛** - 两只眼睛都面向前方或接近前方，至少有一只眼睛/瞳孔是清晰的。

•**脸部** - 脸部相当清晰，面向前方或近前方。

•**接近** - 单个宠物占据了照片的很大一部分（大约超过照片宽度或高度的50%）。

•**动作** **-** 宠物处于动作中（例如，跳跃）。

•**附件** - 伴随的物理或数字附件/道具（如玩具、数字贴纸），不包括项圈和皮带。

•**团体** - 照片中超过1只宠物。

•**拼贴** -数字修饰过的照片（即有数字相框，多张照片的组合）。

•**人** - 照片中的人。

•**遮挡** - 特定的不良物体遮挡了宠物的一部分（即人、笼子或栅栏）。注意，不是所有的遮挡物体都被认为是遮挡物。

•**信息** - 自定义添加的文本或标签（即宠物名称、描述）。

•**模糊** - 明显失焦或嘈杂，特别是宠物的眼睛和脸部。对于模糊条目，"眼睛 "栏总是设置为0。




## 解决方案思路
本次竞赛我们的方案策略如下

- 本次竞赛需要我们预测宠物的受欢迎程度：Pawpularity，label的分布区间是0-100，本身是一个回归问题，我们将其归一化后，将其转化成0-1区间的分布，使其变成分类问题。
- 根据label的分布，做分箱bins，并根据bins来做StratifiedKFold。
- 图像处理方面，我们删掉了一些重复图像，然后做了一些轻度的数据增强和轻度的tta。
- 模型上，我们选择了swin transformer，不同版本的融合，并且我们输出了最后一个feature map，以供之后做SVR。
- 因为本次比赛fastai表现出色，所以参数选择上，我们尽量靠近fastai的一些设置。

方案参考自两篇notebook：

1. [[train]Pytorch Swin+5Fold+some tips | Kaggle](https://www.kaggle.com/ytakayama/train-pytorch-swin-5fold-some-tips)
2. [RAPIDS SVR Boost - 17.8 | Kaggle](https://www.kaggle.com/cdeotte/rapids-svr-boost-17-8)





## 比赛上分历程

1. 使用 [train]Pytorch Swin+5Fold+some tips 公开kernel，并使用5Fold，Public LB : 18.09
2. 调整分箱bins数量，Public LB : 17.93
3. 删掉重复数据，并调整了数据增强，Public LB : 17.89
4. 换成smoothing loss，smooth系数0.1，Public LB : 17.85
5. Inference阶段使用SVR，并与Swin Transformer融合，Public LB : 17.82
6. 在loss函数中，减少label趋向于的0或者100的极值的权重，Public LB : 17.80
7. 调整finetune的 MIN_SIZE_TRAIN 和 MIN_SIZE_TEST，Public LB : 0.311
8. 使用10Fold，Public LB : 17.76
9. 使用多个Swin Transformer的版本（224和384），并融合。Public LB : 17.73
10. 对学习率、epoch、scheduler和tta次数进行简单调参。Public LB : 17.71



## 参数代码

#### Pretrain



## 数据增强

```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB
IMAGENET_STD = [0.229, 0.224, 0.225]  # RGB
def get_train_transforms(epoch, dim = Config.im_size):
    return A.Compose(
        [             
            # resize like Resize in fastai
            A.SmallestMaxSize(max_size=dim, p=1.0),
            A.RandomCrop(height=dim, width=dim, p=1.0),
            A.VerticalFlip(p = 0.5),
            A.HorizontalFlip(p = 0.5)
            #A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ]
  )

def get_inference_fixed_transforms(mode=0, dim = Config.im_size):
    if mode == 0: # do not original aspects, colors and angles
        return A.Compose([
                A.SmallestMaxSize(max_size=dim, p=1.0),
                A.CenterCrop(height=dim, width=dim, p=1.0),
                #A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ], p=1.0)
    elif mode == 1:
        return A.Compose([
                A.SmallestMaxSize(max_size=dim, p=1.0),
                A.CenterCrop(height=dim, width=dim, p=1.0),
                #A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),,
                A.VerticalFlip(p = 1.0)
            ], p=1.0)    
    elif mode == 2:
        return A.Compose([
                A.SmallestMaxSize(max_size=dim, p=1.0),
                A.CenterCrop(height=dim, width=dim, p=1.0),
                #A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                A.HorizontalFlip(p = 1.0)
            ], p=1.0)
    elif mode == 3:
        return A.Compose([
                A.SmallestMaxSize(max_size=dim, p=1.0),
                A.CenterCrop(height=dim, width=dim, p=1.0),
                #A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                A.Transpose(p=1.0)
            ], p=1.0)
        
def get_inference_random_transforms(mode=0, dim = Config.im_size):
    if mode == 0: # do not original aspects, colors and angles
        return A.Compose([
                A.SmallestMaxSize(max_size=dim, p=1.0),
                A.CenterCrop(height=dim, width=dim, p=1.0),
                #A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ], p=1.0)
    else:
        return A.Compose(
            [            
                A.SmallestMaxSize(max_size=dim, p=1.0),
                A.CenterCrop(height=dim, width=dim, p=1.0),
                A.VerticalFlip(p = 0.5),
                A.HorizontalFlip(p = 0.5)
                #A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ]
      ) 
```



## 模型代码

```Python
class PetNet(nn.Module):
    def __init__(
        self,
        model_name = Config.model_path,
        out_features = Config.out_features,
        inp_channels=Config.inp_channels,
        pretrained=Config.pretrained
    ):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, in_chans=inp_channels, num_classes = out_features)
        print("self.model.head.in_features:",self.model.head.in_features)
        self.model.head = nn.Linear(self.model.head.in_features, 128) # 1536
        self.dropout1 = nn.Dropout(0.1)
        self.dense1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(64, 1)
        
    def forward(self, image):
        x1 = self.model(image)          # [bs, 128]
        x = self.dropout1(x1)           # [bs, 128]
        x = self.dense1(x)              # [bs, 64]
        x = self.relu(x)                # [bs, 64]
        x = self.dense2(x)              # [bs, 1]
        x2 = torch.cat([x, x1], dim=1)  # [bs, 129]
        return x, x2
```



## 代码、数据集

+ 代码
  + petfinder train.ipynb
  + perfinder inference.ipynb
+ 数据集
  - 官网图片数据：[PetFinder.my - Pawpularity Contest | Kaggle](https://www.kaggle.com/c/petfinder-pawpularity-score/data)

## TL;DR

竞赛是由petfinder.my举办的，参赛者将通过宠物照片和照片元数据，预测宠物在网络上的 "受欢迎程度"。本次竞赛中我们团队选择今年最热的 Swin Transformer，原因是该模型可以比其他CNN模型更好的看到照片全局的信息。因为本次比赛数据噪音较大，所以我们采用了smoothing loss起到了不错的效果。并且我们输出了模型的最后一个feature map的Image Embedding，将其与元数据结合，利用SVR直接预测，将 Swin Transformer 和 SVR的预测融合后，我们取得了Private LB: 16.9 (Top1%) 的成绩。

