# 附录

本部分对torchvision中自带图像预处理方法进行了分类总结和介绍，主要涉及裁剪、反转和旋转、其他图像变换等19种方法：

[官方torchvision.transforms讲解](https://pytorch.org/docs/stable/torchvision/transforms.html)

### 1.裁剪

**（1）中心裁剪：transforms.CenterCrop**

> *CLASS* `torchvision.transforms.CenterCrop(size)`
>
> ```
> 根据给定的size从从中心进行裁剪
> ```
>
> > **参数：**
> >
> > **size**(*sequence* *or* *int*) - 裁剪后的输出尺寸。若为*sequence*，表示(h, w)；若为*int*，表示(size, size)。

**（2）随机裁剪：transforms.RandomCrop**

> *CLASS* `torchvision.transforms.RandomCrop(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant')`
>
> 	根据给定的size在随机点进行裁剪
>
> > **参数：**
> >
> > **size**(*sequence* *or* *int*) - 裁剪后的输出尺寸。若为*sequence*，表示(h, w)；若为*int*，表示(size, size)。
> >
> > **padding**(*int* *or* *sequence*, *optional*) - 图像填充像素的个数。默认*None*，不填充；若为*int*，图像上下左右均填充int个像素；若为*sequence*，有两个给定值时，第一个数表示左右填充像素个数，第二个数表示上下像素填充个数，有四个给定值时，分别表示左上右下填充像素个数。
> >
> > **fill** - 只针对constant填充模式，填充的具体值。默认为0。若为*int*，各通道均填充该值；若为长度3的tuple时，表示RGB各通道填充的值。
> >
> > **padding_mode** - 填充模式。● constant：特定的常量填充；● edge：图像边缘的值填充● reflect；● symmetric。

**（3）随机长宽比裁剪 ：transforms.RandomResizedCrop**

> *CLASS* `torchvision.transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2)`
>
> 	根据随机大小和长宽比裁剪，并且最后将裁剪的图像resize为给定的size。通常用于训练Inception网络。
>
> >  **参数：**
> >
> > **size** - 期望输出的图像尺寸
> >
> > **scale** - 随机裁剪的区间，默认(0.08, 1.0)，表示随机裁剪的图片在0.08倍到1.0倍之间。
> >
> > **ratio** - 随机长宽比的区间，默认(3/4, 4/3)。
> >
> > **interpolation** - 差值方法，默认为PIL.Image.BILINEAR（双线性差值）

**（4）五分图像：transforms.FiveCrop**

> *CLASS* `torchvision.transforms.FiveCrop(size)`
>
> 	对图像四个角和中心进行裁剪得到五张图像
>
> > **参数：**
> >
> > **size**(*sequence* *or* *int*) - 裁剪后的输出尺寸。若为*sequence*，表示(h, w)；若为*int*，表示(size, size)。

**（5）十分图像：transforms.TenCrop**

> *CLASS* `torchvision.transforms.TenCrop(size, vertical_flip=False)`
>
> 	对图片进行上下左右以及中心裁剪，然后全部翻转（水平或者垂直），获得10张图片
>
> > **参数：**
> >
> > **size**(*sequence* *or* *int*) - 裁剪后的输出尺寸。若为*sequence*，表示(h, w)；若为*int*，表示(size, size)
> >
> > **vertical_flip**(*bool*) - 默认False，水平翻转；否则垂直翻转。

### 2.翻转和旋转    

**（1）依概率水平翻转transforms.RandomHorizontalFlip**

> *CLASS* `torchvision.transforms.RandomHorizontalFlip(p=0.5)`
>
> 	根据给定的概率p水平翻转图像（PIL图像或Tensor）
>
> > **参数：**
> >
> > **p**(*float*) - 翻转概率，默认0.5。

**（2）依概率垂直翻转transforms.RandomVerticalFlip**

> *CLASS* `torchvision.transforms.RandomVerticalFlip(p=0.5)`
>
> 	根据给定的概率p垂直翻转图像（PIL图像或Tensor）
>
> > **参数：**
> >
> > **p**(*float*) - 翻转概率，默认0.5。

**（3）随机旋转：transforms.RandomRotation**

> *CLASS* `torchvision.transforms.RandomRotation(degrees, resample=False, expand=False, center=None, fill=None)`
>
> 	根据*degrees*随机旋转图像一定角度
>
> > **参数：**
> >
> > **degrees**(*sequence* *or* *float* *or* *int*) - 待选择旋转度数的范围。如果是一个数字，表示在(-degrees, +degrees)范围内随机旋转；如果是类似(min, max)的sequence，则表示在指定的最小和最大角度范围内随即旋转。
> >
> > **resample**(*{PIL.Image.NEAREST*, *PIL.Image.BILINEAR*, *PIL.Image.BICUBIC}*, *optional*) - 重采样方式，可选。
> >
> > **expand**(*bool*, *optional*) - 图像尺寸是否根据旋转后的图像进行扩展，可选。若为True，扩展输出图像大小以容纳整个旋转后的图像；若为False或忽略，则输出图像大小和输入图像的大小相同。
> >
> > **center**(*2-tuple*, *optional*) - 旋转中心，可选为中心旋转或左上角点旋转。
> >
> > **fill**(*n-tuple* *or* *int* *or* *float*) - 旋转图像外部区域像素的填充值。此选项仅使用pillow >= 5.2.0。

### 3. 其他图像变换

**（1）转为tensor：transforms.ToTensor**

> *CLASS* `torchvision.transforms.ToTensor`
>
> 	将PIL Image或范围在[0, 255]的numpy.ndarray(H×W×C)转换成范围为[0.0, 1.0]的torch.Float(C×H×W)类型的tensor

**（2）转为PILImage：transforms.ToPILImage**

> *CLASS* `torchvision.transforms.ToPILImage(mode=None)`
>
> 	将tensor(C×H×W)或者numpy.ndarray(H×W×C)的数据转换为PIL Image类型数据，同时保留值范围
>
> > **参数：**
> >
> > **mode**(PIL.Image mode) - 输入数据的颜色空间和像素深度。如果为None(默认)时，会对数据做如下假定：输入为1通道，mode根据数据类型确定；输入为2通道，mode为LA；输入为3通道，mode为RGB；输入为4通道，mode为RGBA。

**（3）填充：transforms.Pad**

> *CLASS* `torchvision.transforms.Pad(padding, fill=0, padding_mode='constant')`
>
> 	对给定的PIL Image使用给定的填充值进行填充
>
> > **参数：**
> >
> > **padding**(*int or tuple*) - 图像填充像素的个数。若为*int*，图像上下左右均填充*int*个像素；若为*tuple*，有两个给定值时，第一个数表示左右填充像素个数，第二个数表示上下像素填充个数，有四个给定值时，分别表示左上右下填充像素个数。
> >
> > **fill** - 只针对constant填充模式，填充的具体值。默认为0。若为*int*，各通道均填充该值；若为长度3的tuple时，表示RGB各通道填充的值。
> >
> > **padding_mode** - 填充模式。● constant：特定的常量填充；● edge：图像边缘的值填充● reflect；● symmetric。

**（4）resize：transforms.Resize**

>*CLASS* `torchvision.transforms.Resize(size, interpolation=2)`
>
>	重置PIL Image的size
>
>> **参数：**
>>
>> **size**(*sequence or int*) - 需求的输出图像尺寸。如果size是类似(h, w)的*sequence*，表示输出图像高为h，宽为w；如果为*int*，则匹配图像较小的边到size，并保持高宽比，如 height > width，图像将被重置为(size * height / width, size)。
>>
>> **interpolation**(*int*, *optional*) - 差值方式，默认为`PIL.Image.BILINEAR`

**（5）标准化：transforms.Normalize**

>*CLASS* `torchvision.transforms.Normalize(mean, std, inplace=False)`
>
>	对tensor image进行标准化。根据给定的n个通道的均值`(mean[1],...,mean[n])`和标准差`(std[1],..,std[n])`计算每个通道的输出值`output[channel] = (input[channel] - mean[channel]) / std[channel]`
>
>> **参数：**
>>
>> **mean**(*sequence*) - 含有每个通道均值的*sequence*
>>
>> **std**(*sequence*) - 含有每个通道标准差的*sequence*
>>
>> **inplace** (*bool*, *optional*) - 是否替换原始数据

**（6）修改亮度、对比度和饱和度：transforms.ColorJitter**

>*CLASS* `torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)`
>
>	随机更改图像的亮度、对比度和饱和度。

**（7）转为灰度图：transforms.Grayscale**

>*CLASS* `torchvision.transforms.Grayscale(num_output_channels=1)`
>
>	将图片转成灰度图
>
>> **参数：**
>>
>> **num_output_channels**(*int*) - （1或3）输出图像的通道数。如果为1，输出单通道灰度图；如果为3，输出3通道，且有r == g == b。

**（8）依概率转为灰度图：transforms.RandomGrayscale**

>*CLASS* `torchvision.transforms.RandomGrayscale(p=0.1)`
>
>	根据概率随机将图片转换成灰度图
>
>> **参数：**
>>
>> **p** (*float*) - 图像转换成灰度图的概率。

**（9）线性变换：transforms.LinearTransformation**

>*CLASS* `torchvision.transforms.LinearTransformation(transformation_matrix, mean_vector)`
>
>	对tensor image做线性变换，可用于白化处理。
>
>> **参数：**
>>
>> **transformation_matrix**(*Tensor*) - tensor [D x D], D = C x H x W
>>
>> **mean_vector**(*Tensor*) - tensor [D], D = C x H x W

**（10）仿射变换：transforms.RandomAffine**

>*CLASS* `torchvision.transforms.RandomAffine(degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0)`
>
>	保持图像中心不变的随机仿射变换

**（11）自定义变换：transforms.Lambda**

>*CLASS* `torchvision.transforms.Lambda(lambd)`
>
>	将自定义的函数(lambda)应用于图像变换
>
>> **参数：**
>>
>> **lambd**(*lambda*) - 用于图像变换的lambda/自定义函数