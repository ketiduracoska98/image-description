from pycocotools.coco import COCO
import json

coco = COCO('annotations/captions_val2017.json')

annotations = coco.loadAnns(coco.getAnnIds())

captions_and_images = []

for annotation in annotations:
    image_id = annotation['image_id']
    caption = annotation['caption']

    image_info = coco.loadImgs(image_id)[0]
    image_name = image_info['file_name']

    captions_and_images.append({
        'image_name': image_name,
        'caption': caption
    })

print(captions_and_images)

with open('captions_and_images.json', 'w') as f:
    json.dump(captions_and_images, f, indent=4)
