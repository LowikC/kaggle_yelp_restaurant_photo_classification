# Notes on Yelp challenge

1. some training images miss associated business id.
2. labels are almost evenly distributed
3. there are strong correlation between some of the labels
4. attribute3 (outdoor_seating) seems hard to predict (~0.54 accuracy)
  1. that's weird, it seems easy to distinguish outdoor seating on pictures.
  2. others attributes should be harder to detect (takes_reservations)
5. TODO: check if attribute3 has been correctly encoded
6. TODO: check if attribute3 can be predicted knowing the other attributes.
  1. Tried quickly with a RF: accuracy is quite low (~0.54)
 