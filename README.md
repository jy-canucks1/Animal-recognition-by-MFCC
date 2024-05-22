# Animal-recognition-by-MFCC

Part 1 : Analyze and preprocess MFCC

Part 2 : Train Animal Sound Recognition Models

## Dataset

* The dataset is from: [https://github.com/YashNita/Animal-Sound-Dataset](https://github.com/YashNita/Animal-Sound-Dataset)
* Converted Stereo to Mono and then Removed all the stereo files
* Each mfcc is truncated or padded with 0 to keep the same size of mfcc coefficients sample (130, 13)
* Normalized with means and standard deviations of each of 13 mfcc coefficients

Dataset summary:

![K-093](https://github.com/jy-canucks1/Animal-recognition-by-MFCC/assets/84373345/3ea98d49-24d7-4f5f-b32a-eb2c9f89f44c)

MFCC visualization examples:

![EDA1](https://github.com/jy-canucks1/Animal-recognition-by-MFCC/assets/84373345/fb3f6ae1-d145-4f18-99bb-5a34b158d07e)

The horizontal pairs are in the same Category (1st row: Cat, 2nd row: Lion, 3rd row: Dog, 4th row: Chicken)


## Model

* Tested 3 models : KNN, AdaBoost, CNN
### KNN (K-Nearest Neighbors)
___

Result:

K = 5

![KNN_5_val](https://github.com/jy-canucks1/Animal-recognition-by-MFCC/assets/84373345/58ff7632-dcc9-4114-8b82-6b7383c9754b)

![KNN_5_test_frame](https://github.com/jy-canucks1/Animal-recognition-by-MFCC/assets/84373345/95f67087-2c19-4176-8b14-6aab518d1253)

![KNN_5_test_file](https://github.com/jy-canucks1/Animal-recognition-by-MFCC/assets/84373345/6e921082-c86e-4aee-a9a0-e9698be4ee3a)

K = 7

![KNN_7_val](https://github.com/jy-canucks1/Animal-recognition-by-MFCC/assets/84373345/273a8506-9688-48df-9a44-1e51ab2826d3)

![KNN_7_test_frame](https://github.com/jy-canucks1/Animal-recognition-by-MFCC/assets/84373345/fb6b9b6a-f1e6-4f0e-bf37-a45424cdaa9a)

![KNN_7_test_file](https://github.com/jy-canucks1/Animal-recognition-by-MFCC/assets/84373345/1efa87fb-37ce-4d73-91b7-22f6bba0deff)

K = 9

![KNN_9_val](https://github.com/jy-canucks1/Animal-recognition-by-MFCC/assets/84373345/55e500e8-2bda-4735-9314-e628e03bafa2)

![KNN_9_test_frame](https://github.com/jy-canucks1/Animal-recognition-by-MFCC/assets/84373345/dc47cb43-83bc-4f6c-88e5-d3d1fe2c8c3e)

![KNN_9_test_file](https://github.com/jy-canucks1/Animal-recognition-by-MFCC/assets/84373345/1a6bdb6f-529f-4468-99f1-fc8fb8af5813)

(K >= 11 was not denoted since all the accuracies were lower than K = 9)

### AdaBoost
___

![AdaBoost_val](https://github.com/jy-canucks1/Animal-recognition-by-MFCC/assets/84373345/cc7187e9-616c-4e27-b0bf-14e8734682b6)

![AdaBoost_test_frame](https://github.com/jy-canucks1/Animal-recognition-by-MFCC/assets/84373345/66dae590-d370-4f6c-955b-9b2360a1f09f)

![AdaBoost_test_file](https://github.com/jy-canucks1/Animal-recognition-by-MFCC/assets/84373345/4324e88d-3bc8-4c7d-89ac-94ecadd31849)


### CNN (Convolutional Neural Network)
___

![CNN_training_progress](https://github.com/jy-canucks1/Animal-recognition-by-MFCC/assets/84373345/5e38d246-0e6e-464a-a1de-27b00616cea2)

![CNN_test](https://github.com/jy-canucks1/Animal-recognition-by-MFCC/assets/84373345/29c4e498-81a4-45f0-9eb7-f0401b5bfe9f)


## Conclusion

### Why all models were not accurate? (All accuracies < 90%)
___

* The dataset was imbalanced (The three most accurate classes had 200 audio files, and some classes had less than 50 files)
* MFCC is well-affected by noises in sound samples.
* There were many samples having different total frame lengths

### KNN wins, CNN is the second, and AdaBoost was the last
___

* CNN might have had better result if the total frame length was constant for all audio files. Some mfcc specturm images have been cut out during preprocessing.
* AdaBoost did not work well with imbalanced dataset. (For only classes with 200 files, it worked relatively well.)

## References
___

What is MFCC? : [https://medium.com/prathena/the-dummys-guide-to-mfcc-aceab2450fd](https://medium.com/prathena/the-dummys-guide-to-mfcc-aceab2450fd)

How to get MFCC? : [https://www.mathworks.com/help/audio/ref/mfcc.html](https://www.mathworks.com/help/audio/ref/mfcc.html)
