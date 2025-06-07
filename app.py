import json
from flask import Flask, request, render_template, jsonify, send_from_directory, redirect, url_for
import os
from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel, BlipProcessor, BlipForConditionalGeneration, pipeline
from PIL import Image
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

from kafka import KafkaProducer,KafkaConsumer
from prometheus_client import start_http_server, Gauge, Counter, Histogram
import time
from werkzeug.utils import secure_filename


start_http_server(8000)
inference_time = Histogram('caption_inference_duration_seconds', 'Time to generate a caption', ['model'])
bleu_score = Gauge('caption_bleu_score', 'BLEU score of model', ['model', 'image'])
rouge_score = Gauge('caption_rouge_l_score', 'ROUGE-L score of model', ['model', 'image'])
cosine_score = Gauge('caption_cosine_similarity', 'Cosine similarity vs COCO', ['model', 'image'])
equivalence_flag = Gauge('caption_equivalent_to_coco', 'Semantic match with COCO (1/0)', ['model', 'image'])
inference_counter = Counter('caption_total_inferences', 'Total inferences run', ['model'])
error_counter = Counter('caption_errors_total', 'Errors during captioning', ['model'])


producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

consumer = KafkaConsumer(
    'image-topic',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    group_id='caption-group',
    auto_offset_reset='earliest'
)

scored_consumer = KafkaConsumer(
    'captions-scored',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    group_id='flask-scored-consumer',
    auto_offset_reset='earliest',
    enable_auto_commit=True
)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

vit_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
vit_tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
vit_image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

caption_generator_git = pipeline("image-to-text", model="microsoft/git-large-coco")

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
            #     producer.send('imag`e-topic', {'image_path': file_path})
            #     producer.flush()  # Ensure the message is sent
            #     return render_template("index.html", message="Image path sent to Kafka!")
                return generate_captions({"image_path": file_path})
            else:
                return render_template("index.html", error="File not found in dataset", dataset_images=load_dataset())

        return render_template("index.html", error="Invalid selection", dataset_images=load_dataset())

    dataset_images = load_dataset()
    return render_template("index.html", dataset_images=dataset_images)

def listen_to_kafka():
    for message in consumer:
        image_path = message.value['image_path']
        caption = generate_captions({"image_path": image_path})
        print(caption)


# Run Kafka consumer in a background thread
def start_kafka_listener():
    listener_thread = threading.Thread(target=listen_to_kafka)
    listener_thread.daemon = True  # Allows the app to exit even if this thread is running
    listener_thread.start()


def get_scored_results(image_path, timeout_secs=10):
    from kafka import KafkaConsumer
    import time
    import json
    image_name = os.path.basename(image_path)
    deadline = time.time() + timeout_secs

    consumer = KafkaConsumer(
        'captions-scored',
        bootstrap_servers='localhost:9092',
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        group_id=None,  # no group ID to avoid committing offsets
        auto_offset_reset='earliest',
        consumer_timeout_ms=1000  # exit loop after a bit if no messages
    )

    while time.time() < deadline:
        for msg in consumer:
            try:
                data = msg.value
                if isinstance(data, list):
                    if data[0].get("image") == image_path or data[0].get("image") == image_name:
                        consumer.close()
                        return data
            except Exception as e:
                print(f"Error parsing scored result: {e}")
        time.sleep(1)

    consumer.close()
    return []



@app.route("/generate_captions", methods=["POST"])
def generate_captions(data=None):
    global flink_results
    if not data:
        data = request.json

    image_path = data.get("image_path")
    if not image_path or not os.path.exists(image_path):
        return jsonify({"status": "error", "message": "Invalid image path"})

    try:
        image = Image.open(image_path)

        # Generate caption using ViT-GPT2 model
        model_name = "ViT-GPT2"
        inference_counter.labels(model=model_name).inc()
        start = time.time()
        try:
            pixel_values = vit_image_processor(image, return_tensors="pt").pixel_values
            generated_ids = vit_model.generate(pixel_values)
            vit_caption = vit_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        except Exception:
            error_counter.labels(model=model_name).inc()
            vit_caption = ""
        finally:
            inference_time.labels(model=model_name).observe(time.time() - start)

        # GIT
        model_name = "GIT"
        inference_counter.labels(model=model_name).inc()
        start = time.time()
        try:
            git_caption = caption_generator_git(image)[0]['generated_text']
        except Exception:
            error_counter.labels(model=model_name).inc()
            git_caption = ""
        finally:
            inference_time.labels(model=model_name).observe(time.time() - start)

        # BLIP
        model_name = "BLIP"
        inference_counter.labels(model=model_name).inc()
        start = time.time()
        try:
            blip_caption = generate_captions_blip(image)
        except Exception:
            error_counter.labels(model=model_name).inc()
            blip_caption = ""
        finally:
            inference_time.labels(model=model_name).observe(time.time() - start)

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
            "BLIP": blip_caption
        }
        if coco_caption:
            captions["COCO"] = coco_caption

        caption_colors = {}
        for model, caption in captions.items():
            if model != "COCO":
                is_equivalent = are_sentences_equivalent(caption, coco_caption if coco_caption else "")
                caption_colors[model] = "green" if is_equivalent else "red"
            # Cosine similarity
            try:
                embedding1 = get_sentence_embedding(caption)
                embedding2 = get_sentence_embedding(coco_caption)
                sim = cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]
                cosine_score.labels(model=model, image=image_name).set(sim)
            except:
                pass

            equivalence_flag.labels(model=model, image=image_name).set(1 if is_equivalent else 0)

        caption_data = {
            "image": image_path,
            "captions": captions
        }


        print(f"Sending data to Kafka: {caption_data}")

        producer.send('captions-input', value=caption_data)
        producer.flush()

        flink_scores = get_scored_results(image_path)

        return render_template("result.html", image_path=image_path, captions=captions, caption_colors=caption_colors, flink_scores=flink_scores)


    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route("/upload_and_caption", methods=["POST"])
def upload_and_caption():
    if "image" not in request.files or "user_caption" not in request.form:
        return jsonify({"status": "error", "message": "Image file and caption are required."})

    file = request.files["image"]
    user_caption = request.form["user_caption"]

    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"status": "error", "message": "Invalid file format."})

    try:
        filename = file.filename
        save_path = os.path.join(DATASET_FOLDER, filename)
        file.save(save_path)

        image = Image.open(save_path)

        # Reuse model inference logic from `generate_captions()`, with user_caption replacing coco_caption
        captions = {}
        caption_colors = {}

        # ViT-GPT2
        model_name = "ViT-GPT2"
        inference_counter.labels(model=model_name).inc()
        start = time.time()
        try:
            pixel_values = vit_image_processor(image, return_tensors="pt").pixel_values
            generated_ids = vit_model.generate(pixel_values)
            vit_caption = vit_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        except Exception:
            error_counter.labels(model=model_name).inc()
            vit_caption = ""
        finally:
            inference_time.labels(model=model_name).observe(time.time() - start)
        captions["ViT-GPT2"] = vit_caption

        # GIT
        model_name = "GIT"
        inference_counter.labels(model=model_name).inc()
        start = time.time()
        try:
            git_caption = caption_generator_git(image)[0]['generated_text']
        except Exception:
            error_counter.labels(model=model_name).inc()
            git_caption = ""
        finally:
            inference_time.labels(model=model_name).observe(time.time() - start)
        captions["GIT-large-COCO"] = git_caption

        # BLIP
        model_name = "BLIP"
        inference_counter.labels(model=model_name).inc()
        start = time.time()
        try:
            blip_caption = generate_captions_blip(image)
        except Exception:
            error_counter.labels(model=model_name).inc()
            blip_caption = ""
        finally:
            inference_time.labels(model=model_name).observe(time.time() - start)
        captions["BLIP"] = blip_caption

        image_name = os.path.basename(save_path)

        for model, caption in captions.items():
            is_equivalent = are_sentences_equivalent(caption, user_caption)
            caption_colors[model] = "green" if is_equivalent else "red"

            try:
                embedding1 = get_sentence_embedding(caption)
                embedding2 = get_sentence_embedding(user_caption)
                sim = cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]
                cosine_score.labels(model=model, image=image_name).set(sim)
            except:
                pass

            equivalence_flag.labels(model=model, image=image_name).set(1 if is_equivalent else 0)

        caption_data = {
            "image": save_path,
            "captions": captions,
            "user_caption": user_caption
        }

        producer.send('captions-input', value=caption_data)
        producer.flush()

        flink_scores = get_scored_results(save_path)

        return render_template(
            "result.html",
            image_path=save_path,
            captions=captions,
            caption_colors=caption_colors,
            flink_scores=flink_scores,
            user_caption=user_caption
        )

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/own-data', methods=['GET', 'POST'])
def own_data():
    if request.method == 'POST':
        user_image = request.files.get('user_image')
        description = request.form.get('description')

        if not user_image or not description:
            return render_template('own_data.html', error="Both image and description are required.")

        filename = secure_filename(user_image.filename)
        upload_dir = os.path.join("static", "uploads")
        os.makedirs(upload_dir, exist_ok=True)

        image_path = os.path.join(upload_dir, filename)
        user_image.save(image_path)

        try:
            image = Image.open(image_path)
            captions = {}
            caption_colors = {}

            # ViT-GPT2
            model_name = "ViT-GPT2"
            inference_counter.labels(model=model_name).inc()
            start = time.time()
            try:
                pixel_values = vit_image_processor(image, return_tensors="pt").pixel_values
                generated_ids = vit_model.generate(pixel_values)
                vit_caption = vit_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            except Exception:
                error_counter.labels(model=model_name).inc()
                vit_caption = ""
            finally:
                inference_time.labels(model=model_name).observe(time.time() - start)
            captions["ViT-GPT2"] = vit_caption

            # GIT
            model_name = "GIT"
            inference_counter.labels(model=model_name).inc()
            start = time.time()
            try:
                git_caption = caption_generator_git(image)[0]['generated_text']
            except Exception:
                error_counter.labels(model=model_name).inc()
                git_caption = ""
            finally:
                inference_time.labels(model=model_name).observe(time.time() - start)
            captions["GIT-large-COCO"] = git_caption

            # BLIP
            model_name = "BLIP"
            inference_counter.labels(model=model_name).inc()
            start = time.time()
            try:
                blip_caption = generate_captions_blip(image)
            except Exception:
                error_counter.labels(model=model_name).inc()
                blip_caption = ""
            finally:
                inference_time.labels(model=model_name).observe(time.time() - start)
            captions["BLIP"] = blip_caption

            image_name = os.path.basename(image_path)

            for model, caption in captions.items():
                is_equivalent = are_sentences_equivalent(caption, description)
                caption_colors[model] = "green" if is_equivalent else "red"

                try:
                    embedding1 = get_sentence_embedding(caption)
                    embedding2 = get_sentence_embedding(description)
                    sim = cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]
                    cosine_score.labels(model=model, image=image_name).set(sim)
                except:
                    pass

                equivalence_flag.labels(model=model, image=image_name).set(1 if is_equivalent else 0)

            captions["COCO"] = description

            # Send to Kafka
            caption_data = {
                "image": image_path,
                "captions": captions,
            }

            producer.send('captions-input', value=caption_data)
            producer.flush()

            # Retrieve Flink scores
            flink_scores = get_scored_results(image_path)

            return render_template(
                "result.html",
                image_path=image_path,
                captions=captions,
                caption_colors=caption_colors,
                flink_scores=flink_scores,
                user_caption=description
            )

        except Exception as e:
            return render_template('own_data.html', error=f"Processing failed: {str(e)}")

    return render_template('own_data.html')


@app.route("/dataset/images")
def dataset_images_popup():
    dataset_images = load_dataset()
    return render_template("popup.html", dataset_images=dataset_images)


@app.route("/dataset/<filename>")
def dataset_file(filename):
    return send_from_directory(DATASET_FOLDER, filename)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)

