# Face Recognition using similar structure of [FaceNet](https://arxiv.org/abs/1503.03832)

Building a face recognition system, to facilitate access to facilities for authorized staff without ID scanning or badges. 


## Face recognition problems commonly fall into two categories:

*  Face Verification - "is this the claimed person?". For example, at some airports, you can pass through customs by letting a system scan  your passport and then verifying that you (the person carrying the passport) are the correct person. A mobile phone that unlocks using your face is also using face verification. This is a **1:1 matching problem.**

*  Face Recognition - "who is this person?". For example, at an office building employees entering without needing to otherwise identify themselves. This is a **1:K matching problem.**


### FaceNet learns a neural network that encodes a face image into a vector of 128 numbers. By comparing two such vectors, you can then determine if two pictures are of the same person.


## 0 - Naive Face Verification

In Face Verification, you're given two images and you have to determine if they are of the same person. The simplest way to do this is to compare the two images pixel-by-pixel. If the distance between the raw images are less than a chosen threshold, it may be the same person!
![](https://github.com/adnaneaabbar/face-recognition-for-authorized-staff/blob/master/assets/pixel_comparison.png?raw=true)

* Of course, this algorithm performs really poorly, since the pixel values change dramatically due to variations in lighting, orientation of the person's face, even minor changes in head position, and so on. 

* You'll see that rather than using the raw image, you can learn an encoding, **f(img)**.  

* By using an encoding for each image, an element-wise comparison produces a more accurate judgement as to whether two pictures are of the same person.

## 1 - Encoding face images into a 128-dimensional vector 

### 1.1 - Using a ConvNet  to compute encodings

The key things you need to know are:

- This network uses 96x96 dimensional RGB images as its input. Specifically, inputs a face image (or batch of **m** face images) as a tensor of shape **(m, n_C, n_H, n_W) = (m, 3, 96, 96)**

- It outputs a matrix of shape **(m, 128)** that encodes each input face image into a 128-dimensional vector

By using a 128-neuron fully connected layer as its last layer, the model ensures that the output is an encoding vector of size 128. You then use the encodings to compare two face images as follows:

![](https://github.com/adnaneaabbar/face-recognition-for-authorized-staff/blob/master/assets/distance_kiank.png?raw=true)

The[Inception](https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202) model is made out of blocks looking like this :
![](http://media5.datahacker.rs/2018/11/Featured-Image-017-CNN-Inception-Network-1.jpg)


So, an encoding is a good one if: 

- The encodings of two images of the same person are quite similar to each other. 

- The encodings of two images of different persons are very different.

The triplet loss function formalizes this, and tries to "push" the encodings of two images of the same person (Anchor and Positive) closer together, while "pulling" the encodings of two images of different persons (Anchor, Negative) further apart. 

![](https://github.com/adnaneaabbar/face-recognition-for-authorized-staff/blob/master/assets/triplet_comparison.png?raw=true)

### 1.2 - The Triplet Loss

For an image **x**, we denote its encoding **f(x)**, where **f** is the function computed by the neural network.

![](https://github.com/adnaneaabbar/face-recognition-for-authorized-staff/blob/master/assets/f_x.png?raw=true)


Training will use triplets of images **(A, P, N)**:  

- A is an "Anchor" image--a picture of a person. 
- P is a "Positive" image--a picture of the same person as the Anchor image.
- N is a "Negative" image--a picture of a different person than the Anchor image.

These triplets are picked from our training dataset. We will write **(A^{(i)}, P^{(i)}, N^{(i)})** to denote the **i**-th training example. 

You'd like to make sure that an image **A^{(i)}** of an individual is closer to the Positive **P^{(i)}** than to the Negative image **N^{(i)}** by at least a margin **alpha**.


## 2 - Loading the pre-trained model

FaceNet is trained by minimizing the triplet loss. But since training requires a lot of data and a lot of computation, we won't train it from scratch here. Instead, we load a previously trained model.

You should add a *weights* folder to your root, and place the weights files in it for this to work. I would love to upload them for you, but it's 67Mo and a total of 226 files. Sorry :( !

Here are some examples of distances between the encodings between three individuals:

![](https://github.com/adnaneaabbar/face-recognition-for-authorized-staff/blob/master/assets/distance_matrix.png?raw=true)

Let's now use this model to perform face verification and face recognition! 

Check the notebook to see the results.

### References:

- Florian Schroff, Dmitry Kalenichenko, James Philbin (2015). [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf)
- Yaniv Taigman, Ming Yang, Marc'Aurelio Ranzato, Lior Wolf (2014). [DeepFace: Closing the gap to human-level performance in face verification](https://research.fb.com/wp-content/uploads/2016/11/deepface-closing-the-gap-to-human-level-performance-in-face-verification.pdf) 
- The pretrained model we use is inspired by Victor Sy Wang's implementation and was loaded using his code: https://github.com/iwantooxxoox/Keras-OpenFace.
- Our implementation also took a lot of inspiration from the official FaceNet github repository: https://github.com/davidsandberg/facenet 
