# Real-Time Depth Estimation
### Introduction
This repository contains a set of python scripts to fine-tune a vgg16 model in order to do real-time depth estimation task

### Network Architecture
I've added a 1*1 conv in order to reduce the number of channels of the last conv layer from 512 to 128.This reduce the model size in order to make it fit in my poor GPU [2GB].

![img_1](./arch.png)

Note: I think implementing a FC-Layers is an improper approach to do this task instead i am currently training a model using a simple Up-Conv technique.
I've added a Scale-Invarient Loss because i think learning relative depth estimation is much easier,It's just a gut feeling.

![img_1](./loss.png)

### Dataset
I've used the NYU Depth V2 dataset.

### Training
For the FC Implementation : I am working on it, but currently i am stucked at 0.15 RMSE on training data and 0.45 RMSE on validation data.

For the Up-Conv Implementation "I will add it's implementation soon in a seprate repo" : I've reached a 0.109 RMSE on Training data and a 0.165 on Validation data. 
