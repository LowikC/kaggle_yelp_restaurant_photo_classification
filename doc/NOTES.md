# Notes on Yelp challenge

1. some training images miss associated business id.
2. labels are almost evenly distributed
3. there are strong correlation between some of the labels
4. attribute3 (outdoor_seating) seems hard to predict (~0.54 accuracy)
  1. that's weird, it seems easy to distinguish outdoor seating on pictures.
  2. others attributes should be harder to detect (takes_reservations)
6. Check if attribute3 can be predicted knowing the other attributes.
  1. Tried quickly with a RF: accuracy is quite low (~0.57)
7. It seems that all attributes (except) can be predicted knowing the other attributes. (acc > 0.8)



## Ideas
* Train multilabel CNN (finetune caffenet) with CrossEntropyLoss
* Extract features from pretrained CNN
  * caffenet, fc6 and fc7
  * googlenet
  * resnet (check license)
* Ensemble of several models
  * Finetuned CNN
  * Prediction on from features (SVM, RF?)
  * Prediction from other labels
    * Predict labels (CNN)
    * for each label, predict it from others labels probability
    * check how to add prior knowledge in a better way
    