Bootcamp Final Proyect

For my end of course project I wanted to develop a convoluted neural network classification model capable of identifying healthy, various nutrient defiencies and pest-affected tomato plant (Solanum lycopersicum) leaves. 

For this purpose I utilised a dataset originally used to diagnose various crop diseases and deficencies in Bangladesh, called OLID-1. Originally, I wanted my project to work with more plant species, but decided to focus on tomato plants because of their popularity and because limiting my dataset was necessary due to processing constraintss.

For the same reasons, i.e processing constraints, I decided to work on Google Colab.

The dataset comprised around 1000 high resolution pictures of healthy, leaf-miner affected, mite affected, jassid and mite affected, potassium(K) defficient, nitrogen(N) defficient and N-K defficient leaves. During loading I resized the images from the original size to 256x256 to comply with processing restraints

It presented an accute class imbalance, with some classes presenting more than 200 samples and others less than 50. To remediate class imbalance, in a first attempt I tried to produce more samples of the minority classes by using image generation, but I created an unbalanced dataset. I then corrected the class imbalance using SMOTE, which yielded a completely even dataset, which all 7 classes having 236 samples.

After coding the labels, I proceeded to build and experiment with 9 different models: 

Model 1
Architecture:
Utilizes Convolutional layers followed by MaxPooling layers for feature extraction.
Starts with 32 filters and gradually increases the number of filters in subsequent layers.
Employs ReLU activation function.
Concludes with a dense layer with 512 units and a softmax output layer.
It had an accuracy of 0.56


Model 2
Architecture:
Similar to Model 1 but with a larger number of filters in the convolutional layers.
Starts with 64 filters and increases to 256 filters.
Employs ReLU activation function.
Concludes with a dense layer with 512 units and a softmax output layer.
It had an accuracy of 0.52

Model 3
Architecture:
Similar to Model 2 but with an increased number of convolutional layers.
Utilizes Convolutional layers followed by MaxPooling layers.
Employs ReLU activation function.
Concludes with a dense layer with 512 units and a softmax output layer.
It had an accuracy of 0.66

Model 4 (Transfer Learning - VGG16)
Architecture:
Utilizes VGG16 pre-trained model as the base with weights from ImageNet.
Adds custom dense layers for fine-tuning.
Employs ReLU activation function in the dense layers.
Utilizes dropout for regularization.
It had an accuracy of 0.77

Model 5 (Transfer Learning - VGG19)
Architecture:
Utilizes VGG19 pre-trained model as the base with weights from ImageNet.
Adds custom dense layers for fine-tuning.
Employs ReLU activation function in the dense layers.
Utilizes dropout for regularization.
It had an accuracy of 0.80

Model 6 (Data Augmentation - VGG19)
Architecture:
Similar to Model 5 but incorporates data augmentation using ImageDataGenerator.
Applies rescaling and additional transformations like rotation, shifting, shearing, zooming, and flipping.
Utilizes VGG19 pre-trained model as the base.
It had an accuracy of 0.35

Model 7 (Data Augmentation - VGG19)
Architecture:
Similar to Model 6 but with additional augmentation parameters.
Utilizes VGG19 pre-trained model as the base.
It had an accuracy of 0.31

Model CNN2
Architecture:
Employs Convolutional layers with Batch Normalization for feature extraction.
Uses MaxPooling layers for downsampling.
Applies ReLU activation function and dropout for regularization.
Utilizes data augmentation for training.
It had an accuracy of 0.10

Model CNN9
Architecture:
Similar to Model CNN2 but without data augmentation.
Employs Convolutional layers with Batch Normalization and MaxPooling.
Concludes with dense layers and softmax output.
It had an accuracy of 0.70

Interestingly, Models CNN2 and CNN9 only differ in the fact that the former uses data augmentation for training, yet they have a very different performance.


Model 5, being based on the VGG19 pre-trained model, benefits from its deeper architecture and hierarchical feature extraction capabilities. This enables it to capture intricate patterns in the images, essential for accurate classification. Additionally, the use of dropout helps mitigate overfitting, enhancing generalization performance. Finally, fine-tuning only the top layers of the pre-trained model allows it to adapt to the specific task while retaining the learned features from ImageNet, making it a powerful choice for transfer learning in image classification tasks.

Model 5 demonstrates respectable performance, particularly in classes 0, 1, 2, and 5, where precision and recall are relatively high. However, it exhibits slightly lower precision and recall values for classes 3, 4, and 6, suggesting potential challenges in classification for these classes.

This can also be explained because leaves, even when affected by a single condition, will never exhibit the exact same manifestations, which difficults identification with a 100% accuracy outside of a lab setting, specially considering that the resolution was diminished to be processing efficient, which gives the model less details that it could learn from. 

In the end, the model can accept and make predictions on images different from the dataset.



