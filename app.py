import json
from flask import Flask, request, render_template, jsonify, send_from_directory
import os
from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel, BlipProcessor, BlipForConditionalGeneration, pipeline
from PIL import Image
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Load the ViT-GPT2 model and tokenizer
vit_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
vit_tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
vit_image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Load the GIT-large-COCO model
caption_generator_git = pipeline("image-to-text", model="microsoft/git-large-coco")

# Load the BLIP processor and model
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Initialize Flask application
app = Flask(__name__)

DATASET_FOLDER = "dataset"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app.config['DATASET_FOLDER'] = DATASET_FOLDER

if not os.path.exists(DATASET_FOLDER):
    os.makedirs(DATASET_FOLDER)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_dataset():
    images = [f for f in os.listdir(DATASET_FOLDER) if allowed_file(f)]
    return images

def generate_captions_blip(image):
    image = image.convert('RGB')
    inputs = blip_processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs, max_new_tokens=50)
    return blip_processor.decode(out[0], skip_special_tokens=True)

def get_sentence_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    sentence_embedding = torch.mean(last_hidden_states, dim=1).squeeze()
    return sentence_embedding

def are_sentences_equivalent(sentence1, sentence2, threshold=0.85):
    embedding1 = get_sentence_embedding(sentence1)
    embedding2 = get_sentence_embedding(sentence2)
    similarity = cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]
    print(f"Similaritatea cosinus între propoziții: {similarity}")
    if similarity > threshold:
        return True
    else:
        return False

@app.route("/", methods=["GET", "POST"])
def upload_or_choose_image():
    if request.method == "POST":
        if "dataset_image" in request.form:
            dataset_image = request.form["dataset_image"]
            file_path = os.path.join(app.config['DATASET_FOLDER'], dataset_image)
            if os.path.exists(file_path):
                return generate_captions({"image_path": file_path})
            else:
                return render_template("index.html", error="File not found in dataset", dataset_images=load_dataset())

        return render_template("index.html", error="Invalid selection", dataset_images=load_dataset())

    dataset_images = load_dataset()
    return render_template("index.html", dataset_images=dataset_images)


@app.route("/generate_captions", methods=["POST"])
def generate_captions(data=None):
    if not data:
        data = request.json

    image_path = data.get("image_path")
    if not image_path or not os.path.exists(image_path):
        return jsonify({"status": "error", "message": "Invalid image path"})

    try:
        image = Image.open(image_path)

        # Generate caption using ViT-GPT2 model
        pixel_values = vit_image_processor(image, return_tensors="pt").pixel_values
        generated_ids = vit_model.generate(pixel_values)
        vit_caption = vit_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Generate caption using GIT-large-COCO model
        git_caption = caption_generator_git(image)[0]['generated_text']

        # Generate caption using BLIP model
        blip_caption = generate_captions_blip(image)

        with open('captions_and_images.json', 'r') as f:
            captions_data = json.load(f)
        image_name = os.path.basename(image_path)  # Extract image name from the path
        coco_caption = None
        for item in captions_data:
            if item["image_name"] == image_name:
                coco_caption = item["caption"]
                break
        captions = {
            "ViT-GPT2": vit_caption,
            "GIT-large-COCO": git_caption,
            "BLIP": blip_caption        }
        if coco_caption:
            captions["COCO"] = coco_caption

        caption_colors = {}
        for model, caption in captions.items():
            if model != "COCO":
                is_equivalent = are_sentences_equivalent(caption, coco_caption if coco_caption else "")
                caption_colors[model] = "green" if is_equivalent else "red"

        return render_template("result.html", image_path=image_path, captions=captions, caption_colors=caption_colors)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route("/dataset/images")
def dataset_images_popup():
    dataset_images = load_dataset()
    return render_template("popup.html", dataset_images=dataset_images)


@app.route("/dataset/<filename>")
def dataset_file(filename):
    return send_from_directory(DATASET_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
