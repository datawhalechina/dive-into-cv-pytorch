# pytorch-vgg-cifar10
This is the PyTorch implementation of VGG network trained on CIFAR10 dataset 

### Requirements. 
[PyTorch] (https://github.com/pytorch/pytorch)

[torchvision] (https://github.com/pytorch/vision)

### Update.
Adding support for CPU. Add ```--cpu``` can make the training or evaluation in cpu mode.

### Download the model
The trained VGG model. 92.4% Accuracy [VGG](http://www.cs.unc.edu/~cyfu/cifar10/model_best.pth.tar)

### Evaluation 
	
	# CUDA
	wget http://www.cs.unc.edu/~cyfu/cifar10/model_best.pth.tar
	python main.py --resume=./model_best.pth.tar -e
	# or use CPU version
	wget http://www.cs.unc.edu/~cyfu/cifar10/model_best_cpu.pth.tar
	python main.py --resume=./model_best_cpu.pth.tar -e --cpu
### Train with script! (16-bit precision) 
	
	
	./run.sh 
	
Using the run.sh script to generate the training log and models of different versions of VGG in 16-bit or 32-bit precision.	
Then use the ipython notebook plot.ipynb to view the results.
	
![alt text](vgg_plot.png)


