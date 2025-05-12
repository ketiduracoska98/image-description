from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import FlinkKafkaConsumer, FlinkKafkaProducer
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.typeinfo import Types
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

def evaluate_caption(data):
    try:
        print(f"[INFO] Received data: {data}")  # 🔍 Log incoming message

        obj = json.loads(data)
        captions = obj.get('captions', {})
        image = obj.get('image', '')

        reference = captions.get("COCO", None)
        if not reference:
            print("[WARN] Missing COCO reference caption.")
            return json.dumps({"error": "Missing COCO reference"})

        results = []
        for model, caption in captions.items():
            if model == "COCO":
                continue

            # Compute BLEU score
            smooth_fn = SmoothingFunction().method1
            bleu = sentence_bleu([reference.split()], caption.split(), smoothing_function=smooth_fn)

            # Compute ROUGE-L score
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            rouge = scorer.score(reference, caption)['rougeL'].fmeasure

            score_entry = {
                "image": image,
                "model": model,
                "BLEU": bleu,
                "ROUGE-L": rouge
            }
            print(f"[INFO] Scored model '{model}': {score_entry}")

            results.append(score_entry)

        return json.dumps(results)

    except Exception as e:
        error_message = f"[ERROR] Failed to evaluate caption: {str(e)}"
        print(error_message)
        return json.dumps({"error": str(e)})


def create_kafka_source(bootstrap_servers, topic, group_id='caption-evaluator-group'):
    consumer_props = {
        'bootstrap.servers': 'kafka:29092',
        'group.id': group_id,
        'auto.offset.reset': 'earliest',
        'fetch.max.wait.ms': '120000',
        'max.poll.interval.ms': '120000',
        'max.request.size': '1048576000'
    }

    source = FlinkKafkaConsumer(
        topics=topic,
        deserialization_schema=SimpleStringSchema(),
        properties=consumer_props
    )
    source.set_start_from_earliest()

    return source


def create_kafka_sink(bootstrap_servers, topic):
    producer_props = {
        'bootstrap.servers': bootstrap_servers
    }

    producer = FlinkKafkaProducer(
        topic=topic,
        serialization_schema=SimpleStringSchema(),
        producer_config=producer_props
    )

    return producer


def main():
    env = StreamExecutionEnvironment.get_execution_environment()
    env.add_jars("file:///opt/flink/lib/flink-sql-connector-kafka-3.0.0-1.17.jar")
    env.set_parallelism(1)

    bootstrap_servers = 'kafka:29092'
    input_topic = 'captions-input'
    output_topic = 'captions-scored'

    print("[INFO] Setting up Kafka source and sink...")
    source = create_kafka_source(bootstrap_servers, input_topic)
    sink = create_kafka_sink(bootstrap_servers, output_topic)

    print("[INFO] Starting stream processing...")
    stream = env.add_source(source).map(evaluate_caption, output_type=Types.STRING())
    stream.add_sink(sink)

    env.execute("Kafka Caption Evaluation Job")


if __name__ == "__main__":
    main()
