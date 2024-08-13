import json

with open("/Users/zhengjiawen/Desktop/datasets/eeg-imagenet/text/label2text.json", "r") as f:
    image2label = json.load(f)

with open("/Users/zhengjiawen/Desktop/datasets/eeg-imagenet/text/image-list.json", "r") as f:
    image_list = json.load(f)


label_list = [image2label[image.split("_")[0]] for image in image_list]

with open("label-list.json", "w") as f:
    json.dump(label_list, f)

