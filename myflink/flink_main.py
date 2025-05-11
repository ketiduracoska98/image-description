import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import FlinkKafkaConsumer, FlinkKafkaProducer
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.typeinfo import Types


def evaluate_caption(data):
    """
    Evaluates the BLEU and ROUGE-L scores for the given caption data.
    Assumes data is in JSON format containing 'caption' and 'reference' fields.
    """
    try:
        obj = json.loads(data)
        candidate = obj.get('caption', '').split()
        reference = [obj.get('reference', '').split()]

        # Compute BLEU score
        smooth_fn = SmoothingFunction().method1
        bleu = sentence_bleu(reference, candidate, smoothing_function=smooth_fn)

        # Compute ROUGE-L score
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge = scorer.score(obj.get('reference', ''), obj.get('caption', ''))['rougeL'].fmeasure

        # Prepare the output
        result = {
            "image": obj.get("image", ""),
            "model": obj.get("model", ""),
            "BLEU": bleu,
            "ROUGE-L": rouge
        }

        return json.dumps(result)

    except Exception as e:
        # Error handling for invalid or missing fields
        return json.dumps({"error": str(e)})


def create_kafka_source(bootstrap_servers, topic, group_id='caption-evaluator-group'):
    """
    Create a Flink Kafka source.
    """
    consumer_props = {
        'bootstrap.servers': bootstrap_servers,
        'group.id': group_id,
        'auto.offset.reset': 'earliest'
    }

    source = FlinkKafkaConsumer(
        topics=topic,
        deserialization_schema=SimpleStringSchema(),
        properties=consumer_props
    )
    source.set_start_from_earliest()  # Start consuming from the earliest offset

    return source


def create_kafka_sink(bootstrap_servers, topic):
    """
    Create a Flink Kafka sink.
    """
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
    # Set up the Flink execution environment
    env = StreamExecutionEnvironment.get_execution_environment()

    # Add the Kafka connector JAR
    env.get_config().get_configuration().set_string(
        "pipeline.jars", "file:///path/to/flink-connector-kafka_2.11-1.18.0.jar"
    )

    env.set_parallelism(1)

    # Kafka source and sink setup
    bootstrap_servers = 'kafka:9092'
    input_topic = 'captions-input'
    output_topic = 'captions-scored'

    source = create_kafka_source(bootstrap_servers, input_topic)
    sink = create_kafka_sink(bootstrap_servers, output_topic)

    # Process the stream
    stream = env.add_source(source).map(evaluate_caption, output_type=Types.STRING())

    # Send the processed stream to the Kafka sink
    stream.add_sink(sink)

    # Execute the Flink job
    env.execute("Kafka Caption Evaluation Job")



if __name__ == "__main__":
    main()
