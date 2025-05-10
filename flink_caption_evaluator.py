import json
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.common.typeinfo import Types
from nltk.translate.bleu_score import SmoothingFunction



def evaluate_caption(data):
    obj = json.loads(data)
    candidate = obj['caption'].split()
    reference = [obj['reference'].split()]

    smooth_fn = SmoothingFunction().method1

    bleu = sentence_bleu(reference, candidate, smoothing_function=smooth_fn)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge = scorer.score(obj['reference'], obj['caption'])['rougeL'].fmeasure

    return json.dumps({
        "image": obj["image"],
        "model": obj["model"],
        "BLEU": bleu,
        "ROUGE-L": rouge
    })

env = StreamExecutionEnvironment.get_execution_environment()
env.set_parallelism(1)

with open("caption_stream.txt", "r") as f:
    input_lines = [line.strip() for line in f if line.strip()]

data_stream = env.from_collection(input_lines, type_info=Types.STRING())

evaluated_stream = data_stream.map(
    evaluate_caption,
    output_type=Types.STRING()
)

evaluated_stream.print()

env.execute("Caption Evaluation Job")

