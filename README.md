## Introduction

This project implements Normalizer-Free ResNets (NFNets), a high-performance class of image classifiers that outperform batch-normalized networks on large-scale datasets like ImageNet, achieving state-of-the-art accuracy without the need for batch normalization. The key contribution of this work is the Adaptive Gradient Clipping (AGC), which allows NFNets to train with large batch sizes and strong data augmentations.

## Paper Summary

Title: High-Performance Large-Scale Image Recognition Without Normalization

Authors: Andrew Brock, Soham De, Samuel L. Smith, Karen Simonyan

Abstract: This paper presents a method to train deep neural networks without batch normalization using Normalizer-Free ResNets. The proposed models achieve state-of-the-art accuracy on ImageNet while being up to 8.7× faster to train. The paper introduces a novel technique called Adaptive Gradient Clipping (AGC) that enables stable training at large batch sizes.

## training 

![Screenshot 2024-09-09 021053](https://github.com/user-attachments/assets/cbaab984-65b8-4159-9cf4-86c1c4100a44)

![Screenshot 2024-09-09 021134](https://github.com/user-attachments/assets/a86303f5-8fa7-40ef-895b-a4a6e54af725)
Summary of NFNet bottleneck block design and architectural differences. 

## Model performance

![Screenshot 2024-09-09 021111](https://github.com/user-attachments/assets/c426f6d9-8976-44d9-aa72-6377ee3297df)


## NFNet transition block

![Screenshot 2024-09-09 021200](https://github.com/user-attachments/assets/0dfdc5de-c65d-415d-9531-5a5ab9bf3f44)
