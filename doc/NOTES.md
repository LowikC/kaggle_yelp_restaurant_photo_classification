# Notes on Yelp challenge

1. some training images miss associated business id.
2. labels are almost evenly distributed
3. there are strong correlation between some of the labels

4. attribute3 (outdoor_seating) seems hard to predict (~0.54 accuracy)
  1. that's weird, it seems easy to distinguish outdoor seating on pictures.
  2. others attributes should be harder to detect (takes_reservations)
  3. My guess is that only some instances in a group are taken outside, so, most of the pictures associated to the business are not taken outside. That means that considering that the business labels correspond to the photos labels is wrong.
  
6. Check if attribute3 can be predicted knowing the other attributes.
  1. Tried quickly with a RF: accuracy is quite low (~0.57)
  
7. It seems that all attributes (except att3) can be predicted knowing the other attributes. (acc > 0.8)

8. Trained a multilabel CNN (finetune caffenet) with CrossEntropyLoss but the validation loss doesn't decrease.
  1. I can't understand why ...
  
9. Trained a multitask CNN
   1. I used two output for each task, and then a Softmax layer. Maybe a single output and then a sigmoid will be better.
   2. Training is done at instance level (see 4.)
   
10. Extracted CNN features from caffenet and googlenet
   
11. Trained independent logistic regression classifier on extracted features
  1. Optimize hyperparameters on f1-score.
  


## Ideas/Todos
* Integrate correlation between attributes
  * for each label, predict it from others labels probability
  * check how to add prior knowledge in a better way
  * a possibility is to trained in cascade, starting with the easier attributes and then add the prediction of previous attributes to the CNN features.
  
* Trained classifier at instance level and merge after
  * Check how to aggregate prediction to business

* Try to separe instances inside a business group
  * Check if similarity measure could help 

* Check multilabel CNN (finetune caffenet) with CrossEntropyLoss

* Extract features from pretrained CNN
  * resnet (check license)
  
* Try other models for prediction from CNN features
  * XGBoost, ...
  
* Ensemble of several models
  * Majority vote
  * Stacking
  * ...


## Links

1. https://github.com/garydoranjr/misvm
  * multi instance svm
  * seems easy to use
2. Deep Multiple Instance Learning for Image Classification and Auto-Annotation: http://jiajunwu.com/papers/dmil_cvpr.pdf

3. Deep Multi-Instance Transfer Learning: http://dkotzias.com/papers/multi-instance%20deep%20learning.pdf

4. DEEP LEARNING OF FEATURE REPRESENTATION WITH MULTIPLE INSTANCE LEARNING FOR MEDICAL IMAGE ANALYSIS: http://research.microsoft.com/pubs/232681/[2014][ICASSP]deep%20learning%20of%20feature%20representation%20with%20multiple%20instance%20for%20medical%20image%20analysis.pdf

5. https://sites.google.com/site/xyzliwen/resource/multiple_instance_learning
