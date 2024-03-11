from detectron2.data.datasets import register_coco_instances
print("--- Registering Redacted Text Detection Datasets ---")
# We register the datasets below so that we can use them in the various training scripts.
register_coco_instances("classic_train", {}, "../resources/dataset/train/classic_train.json", "../resources/dataset/train/images")
register_coco_instances("classic_test", {}, "../resources/dataset/test/classic_test.json", "../resources/dataset/test/images")

# Also add the extended scripts
register_coco_instances("extended_train", {}, "../resources/dataset/train/extended_train.json", "../resources/dataset/train/images")
register_coco_instances("extended_test", {}, "../resources/dataset/test/extended_test.json", "../resources/dataset/test/images")

# Also register the different splits we use for the training data variance experiment
register_coco_instances("train_10", {}, "../resources/dataset/train/train_10.json", "../resources/dataset/train/images")

register_coco_instances("train_20", {}, "../resources/dataset/train/train_20.json", "../resources/dataset/train/images")

register_coco_instances("train_40", {}, "../resources/dataset/train/train_40.json", "../resources/dataset/train/images")

register_coco_instances("train_60", {}, "../resources/dataset/train/train_60.json", "../resources/dataset/train/images")

register_coco_instances("train_80", {}, "../resources/dataset/train/train_80.json", "../resources/dataset/train/images")
