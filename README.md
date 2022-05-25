# Voice-Recognition-using-SVC-KNN-GaussianNB


**1. Problem Formulation**
Using the MLEnd Hums and Whistles dataset, implement a machine learning solution/pipeline to identify the voice label of input audio.

**2. Machine Learning pipeline**
Import raw data:

Data('hum' audio Audio files) of all songs mixed in '.wav' format.

**Data Preparation:**

-Since we are not concerned with attributes such as the interpretation-type(hum/whistle) and song given the purpose of our machine learning pipeline (voice recognition), we have only taken the hum audio files for all the songs and extracted the 'participant_id'.

-Thus we obtained a clean data frame consisting of columns viz. ('file_id','participant').

**Feature engineering (transform raw data into features):**

-The actual audio files in '.wav' format were imported with the help of python module - 'glob'

-Multiple audio features considered to be relevant for audio biometric upon reasearching listed ahead were extracted with our custom function 'getXY()'. (power, pitch_mean, pitch_std, voiced_fr, spectral_centroids, zero_crossings, spectral_rolloff, mfccs)

**Model selection:**

The below 2 models were tested for our dataset since we are dealing with a multiclass classification problem:

1)Support Vector Classifier

2)The k-nearest neighbour

Prediction generation:

A single audio file can be given as input to predict into our created model to detect the participant_id.

**3. Transformation stage**
-The input audio files are being transformed into a set of relevant audio features (power, pitch_mean, pitch_std, voiced_fr, spectral_centroids, zero_crossings, spectral_rolloff, mfccs) using our custom 'getXy function'.

-This is being generated as a 2-D numpy array with 8 columns, each column corresponding to a particular feature.

-The 'Standard Scaler' will then receive the 2 dimensional array of features and it will scale it based on the mean and standard deviation of each feature.

**4. Modelling**
The below 2 models have been considered for our dataset since we are dealing with a multiclass classification problem:

1)Support Vector Classifier :

SVM (Support Vector Machine) classifies the data using hyperplane which acts like a decision boundary between different classes.It tries to find the best and optimal hyperplane which has maximum margin from each Support Vector. It is suitable for classification problems.

2)The k-nearest neighbour classifier :

kNN classifies an observation with the class that the majority of it's k neighbours belong to. Given a new observation(to be predicted) kNN model will calculate the distance to all the training samples.

**5.Methodology**
-The MLEnd H&W dataset is being split into 80% training data and 20% test data used for both training and validation respectively.

-Due to data imbalance across various classes, we have used SMOTE oversampling technique and regenerated the X_train and y_train data.

-The performance of the model is assessed on the basis of it's traing accuracy, validation accuracy.

**6. Dataset**
-Since we are not concerned with attributes such as the interpretation-type(hum/whistle) and song given the purpose of our machine learning pipeline (voice recognition).

-As the pitch of whistle audio files can be misleading for the voice recognition problem, we have only taken the hum audio files for all the songs and extracted the 'participant_id' attribute.

-Thus we obtained a clean data frame consisting of columns viz. ('file_id','participant') being used for further processing.
