"""Script to run T5 predictions on a dataset of templated clozes."""

import functools
import gzip
import json
import t5
from t5.data.preprocessors import noise_span_to_unique_sentinel
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
import tensorflow
import os

tf.disable_eager_execution()

cwd = os.getcwd()
# Input file
JSON_FILE = cwd + "/data/TREx"
REL_FILE = cwd + "/data/relations.jsonl"
TSV_FILE = cwd + "/data/TREx_entity.tsv"
Y_TOK = "Y"
Y_ID = 63

def json_to_tsv(in_fname, out_fname, rel_fname):
  # Read relations
  rel2templ = {}
  with open(rel_fname) as f:
    for line in f:
      item = json.loads(line.strip())
      rel2templ[item["relation"]] = item["template"]
    tf.logging.info("Found %d relation templates.", len(rel2templ))

    count = 0
    for rel in rel2templ:
      in_path = in_fname + "/" + rel + ".jsonl"
      try:
        with tf.io.gfile.GFile(in_path, "r") as infile, tf.io.gfile.GFile(out_fname, "w") as outfile:
          for line in infile:
            item = json.loads(line.strip())
            question = rel2templ[item["predicate_id"]].replace(
                "[X]", item["sub_label"]).replace("[Y]", Y_TOK)
            answer = item["obj_label"]
            # Write this line as <question>\t<answer>
            outfile.write("%s\t%s\n" % (question, answer))
            count += 1
            tf.logging.log_every_n(
                tf.logging.INFO,
                "Wrote %d examples to %s." % (count, out_fname),
                1000)
      except:
        print ("Relation %s not found. Ignored."%(rel))
  return count

# Create TSVs and get counts.
tf.logging.info("Generating TSVs.")
num_examples = json_to_tsv(JSON_FILE, TSV_FILE, REL_FILE)
tf.logging.info("Wrote total %d examples", num_examples)

# Preprocessor for T5.
def preprocessor(ds):
  def normalize_text(text):
    """Lowercase and remove quotes from a TensorFlow string."""
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text,"'(.*)'", r"\1")
    return text

  def to_inputs_and_targets(ex):
    """Map {"question": ..., "answer": ...}->{"inputs": ..., "targets": ...}."""
    return {
        "inputs": normalize_text(ex["question"]),
        "targets": normalize_text(ex["answer"])
    }
  return ds.map(to_inputs_and_targets,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Replace masked tokens with sentinel.
def noise_preprocessor(ds, vocabulary, **unused_kwargs):
  def my_fn(features):
    tokens = features["inputs"]
    noise_mask = tf.equal(tokens, Y_ID)
    # noise_mask = tf.one_hot(features["index"], tf.size(tokens), dtype=tf.bool)
    inputs = noise_span_to_unique_sentinel(tokens, noise_mask, vocabulary)
    targets = features["targets"]
    return {"inputs": inputs, "targets": features["targets"],
            "inputs_plaintext": features["inputs_plaintext"],
            "targets_plaintext": features["targets_plaintext"]}
  return ds.map(my_fn,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Create a task.
t5.data.TaskRegistry.add(
    "lm_as_kb",
    t5.data.TextLineTask,
    split_to_filepattern={"validation": TSV_FILE},
    # Supply a function which preprocesses text from the tf.data.Dataset.
    text_preprocessor=[
        functools.partial(
            t5.data.preprocessors.parse_tsv,
            field_names=["question", "answer"]),
        preprocessor],
    token_preprocessor=[noise_preprocessor],
    # Use the same vocabulary that we used for pre-training.
    sentencepiece_model_path=t5.data.DEFAULT_SPM_PATH,
    # Lowercase targets before computing metrics.
    postprocess_fn=t5.data.postprocessors.lower_text,
    # We'll use accuracy as our evaluation metric.
    metric_fns=[t5.evaluation.metrics.accuracy],
    # Not required, but helps for mixing and auto-caching.
    num_input_examples={"validation": num_examples}
)

# Inspect a few preprocessed examples.
task = t5.data.TaskRegistry.get("lm_as_kb")
ds = task.get_dataset(split="validation", sequence_length={"inputs": 32, "targets": 32})
print("A few preprocessed examples...")
for ex in tfds.as_numpy(ds.take(5)):
  print(ex)

# Setup model.
MODEL_SIZE = "base" #@param["small", "base", "large", "3B", "11B"]
# Public GCS path for T5 pre-trained model checkpoints
BASE_PRETRAINED_DIR = "pretrained_models"
MODELS_DIR = "output"
PRETRAINED_DIR = os.path.join(BASE_PRETRAINED_DIR, MODEL_SIZE)
MODEL_DIR = os.path.join(BASE_PRETRAINED_DIR, MODEL_SIZE)

import gin

with gin.unlock_config():
  gin.parse_config_file(os.path.join(PRETRAINED_DIR, "operative_config.gin"))

# Set parallelism and batch size to fit on v2-8 TPU (if possible).
# Limit number of checkpoints to fit within 5GB (if possible).
model_parallelism, train_batch_size, keep_checkpoint_max = {
    "small": (1, 256, 16),
    "base": (2, 128, 8),
    "large": (8, 64, 4),
    "3B": (8, 16, 1),
    "11B": (8, 16, 1)}[MODEL_SIZE]

tf.io.gfile.makedirs(MODEL_DIR)
# The models from our paper are based on the Mesh Tensorflow Transformer.
model = t5.models.MtfModel(
    model_dir=PRETRAINED_DIR,
    model_parallelism=model_parallelism,
    tpu=None,
    batch_size=train_batch_size,
    sequence_length={"inputs": 32, "targets": 32},
    learning_rate_schedule=0.003,
    save_checkpoints_steps=5000,
    keep_checkpoint_max=None,
    iterations_per_loop=100,
)

# Run evaluation.
model.eval(
    mixture_or_task_name="lm_as_kb",
    checkpoint_steps="all"
)

# Print predictions.
import random

def print_random_predictions(task_name, n=10):
  """Print n predictions from the validation split of a task."""
  # Grab the dataset for this task.
  ds = t5.data.TaskRegistry.get(task_name).get_dataset(
      split="validation",
      sequence_length={"inputs": 32, "targets": 32},
      shuffle=False)

  def _prediction_file_to_ckpt(path):
    """Extract the global step from a prediction filename."""
    return int(path.split("_")[-2])

  # Grab the paths of all logged predictions.
  prediction_files = tf.io.gfile.glob(
      os.path.join(
          MODEL_DIR,
          "validation_eval/%s_*_predictions" % task_name))
  # Get most recent prediction file by sorting by their step.
  latest_prediction_file = sorted(
      prediction_files, key=_prediction_file_to_ckpt)[-1]

  # Collect (inputs, targets, prediction) from the dataset and predictions file
  results = []
  with tf.io.gfile.GFile(latest_prediction_file) as preds:
    for ex, pred in zip(tfds.as_numpy(ds), preds):
      results.append((tf.compat.as_text(ex["inputs_plaintext"]),
                      tf.compat.as_text(ex["targets_plaintext"]),
                      pred.strip()))

  print("<== Random predictions for %s using checkpoint %s ==>\n" %
        (task_name,
         _prediction_file_to_ckpt(latest_prediction_file)))

  for inp, tgt, pred in random.choices(results, k=10):
    print("Input:", inp)
    print("Target:", tgt)
    print("Prediction:", pred)
    print("Counted as Correct?", tgt == pred)
    print()

print_random_predictions("lm_as_kb")
