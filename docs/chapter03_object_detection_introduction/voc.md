

### 4.PASCAL VOC
              
* #### 简介 
                  
PASCAL VOC为图像分类与物体检测提供了一整套标准的的数据集，并从2005年到2012年每年都举行一场图像检测竞>赛。PASCAL全称为Pattern Analysis, Statical Modeling and Computational Learning，其中常用的数据集主要>有VOC2007与VOC2012两个版本，VOC2007中包含了9963张标注过的图片以及24640个物体标签。在VOC2007之上，VOC2012进一步升级了数据集，一共11530张图片，包括人类；动物（鸟、猫、牛、狗、马、羊）；交通工具（飞机、自>行车、船、公共汽车、小轿车、摩托车、火车）；室内（瓶子、椅子、餐桌、盆栽植物、沙发、电视）20个物体类>别，图片尺寸为500x375。VOC整体图像质量较好，标注比较完整，非常适合模型的性能测试，比较适合做基线。     
      
<img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter02/2.1_dataloader_and_augmentation/VOC展示.png">
         
 * ### 数据格式    
 ```
 .
└── VOCdevkit     #根目录
    └── VOC2012   #不同年份的数据集，这是2012的
        ├── Annotations        # 存放xml标签文件，与JPEGImages中的图片一一对应
        │     ├── 00001.xml 
        │     └── 00002.xml 
        ├── ImageSets          
        │   └── Main
        │     ├── train.txt    # txt文件中每一行包含一个图片的名称
        │     └── val.txt
        └── JPEGImages         # 存放源图片
              ├── 00001.jpg     
              └── 00002.jpg     
 ```
 
 * 官网： http://host.robots.ox.ac.uk/pascal/VOC/

