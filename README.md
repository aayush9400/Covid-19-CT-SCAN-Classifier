# Covid-19-Classifier
This is a project with a working website integrated with a CNN model to make predictions whether a patient is covid-19 positive or not
# Training of model
Different models were trained like VGG16, Autoencoder classifier and InceptionV3. In the end InceptionV3 produced the best results.
# Evalutation and results
The dataset was divided into train, validation and test set beforhand to ensure the sanity of further process.

The model was selected while training which had the highest val_accuracy.

# Conclusion
The DenseNet201 model with modified head converged to classifiy the prositive and the negative cases of COVID19 with 99% accuracy, 98% recall (macro avg) and 98% precision (macro avg).

# Further work
The model can be further improved with more COVID19 positive cases for training.
