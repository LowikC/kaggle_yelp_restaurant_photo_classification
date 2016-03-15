# Notes on Yelp challenge

1. some training images miss associated business id.
2. labels are almost evenly distributed
3. there are strong correlation between some of the labels

4. attribute3 (outdoor_seating) seems hard to predict (~0.54 accuracy)
  1. that's weird, it seems easy to distinguish outdoor seating on pictures.
  2. others attributes should be harder to detect (takes_reservations)
  3. My guess is that only some instances in a group are taken outside, so, most of the pictures associated to the business are not taken outside.
  That means that considering that the business labels correspond to the photos labels is wrong.
  
6. Check if attribute3 can be predicted knowing the other attributes.
  1. Tried quickly with a RF: accuracy is quite low (~0.57)
  
7. It seems that all attributes (except att3) can be predicted knowing the other attributes. (acc > 0.8)

8. Trained a multilabel CNN (finetune caffenet) with CrossEntropyLoss but the validation loss doesn't decrease.
  1. I can't understand why ...
  
9. Trained a multitask CNN
   1. I used two output for each task, and then a Softmax layer. Maybe a single output and then a sigmoid will be better.
   2. Training is done at instance level (see 4.)
   
10. Extracted CNN features from caffenet and googlenet
   




## Ideas
* Train multilabel CNN (finetune caffenet) with CrossEntropyLoss
* 
Extract features from pretrained CNN
  * caffenet, fc6 and fc7
  * googlenet
  * resnet (check license)
  
* Ensemble of several models
  * Finetuned CNN
  * Prediction on from features (LRm XGBoost?)
  * Prediction from other labels
    * Predict labels (CNN)
    * for each label, predict it from others labels probability
    * check how to add prior knowledge in a better way
    
* We d