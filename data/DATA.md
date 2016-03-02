# Organization of the data directory

```
data
¦   DATA.md
¦   attribute_id_to_label.json  # The mapping between attributes ID and their labels.
¦
+---raw  # Provided data
    ¦   test_photo_to_biz.csv
    ¦   train_photo_to_biz_ids.csv
    ¦   train.csv
    ¦
    +---train_photos
    ¦   ¦   10.jpg
    ¦   ¦   103.jpg
    ¦   ¦   ...
    ¦
    +---test_photos
    ¦   ¦   10.jpg
    ¦   ¦   103.jpg
    ¦   ¦   ...
```