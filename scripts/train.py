from tqdm import tqdm
import numpy as np
import argparse
import logging
import os
import sys
import tensorflow as tf
from datasets import load_dataset,load_metric
from transformers import AutoTokenizer,TFAutoModelForSeq2SeqLM,DataCollatorForSeq2Seq,create_optimizer
import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize
from transformers.keras_callbacks import PushToHubCallback
from tensorflow.keras.callbacks import TensorBoard as TensorboardCallback

def preprocess_function(examples):
        inputs = tokenizer(
            examples["text"], max_length=max_input_length, truncation=True
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["title"], max_length=max_target_length, truncation=True
            )

        inputs["labels"] = labels["input_ids"]
        return inputs
    
def evaluate(model, dataset):
        all_predictions = []
        all_labels = []
        for batch in tqdm(dataset):
            predictions = model.generate(batch["input_ids"])
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            labels = batch["labels"].numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
            decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
            all_predictions.extend(decoded_preds)
            all_labels.extend(decoded_labels)
            result = metric.compute(
                predictions=decoded_preds, references=decoded_labels, use_stemmer=True
            )
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        return {k: round(v, 4) for k, v in result.items()}
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument("--model_name", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay_rate", type=float, default=0.01)
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])

    args, _ = parser.parse_known_args()

    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    dataset_path="/opt/ml/input/data/dataset"
    max_input_length = 512
    max_target_length = 64
    test_size=.10

    dataset = load_dataset('json', data_files=os.path.join(dataset_path, args.dataset_file_name))
        
    remove_col = ["pub_time","labels"]
    dataset = dataset["train"].remove_columns(remove_col)


    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    processed_dataset = tokenized_datasets.shuffle().train_test_split(test_size=test_size)


    model = TFAutoModelForSeq2SeqLM.from_pretrained(args.model_id)

    if args.fp16:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf")

    train = processed_dataset["train"].to_tf_dataset(
    columns=["input_ids", "attention_mask", "labels"],
    shuffle=True,
    batch_size=args.train_batch_size,
    collate_fn=data_collator)

    test = processed_dataset["test"].to_tf_dataset(
    columns=["input_ids", "attention_mask", "labels"],
    shuffle=True,
    batch_size=args.eval_batch_size,
    collate_fn=data_collator)

    num_train_steps = len(train) * args.num_train_epochs
    optimizer, lr_schedule = create_optimizer(
        init_lr=args.learning_rate,
        num_train_steps=num_train_steps,
        weight_decay_rate=args.weight_decay_rate,
        num_warmup_steps=args.num_warmup_steps,
    )
    model.compile(optimizer=optimizer)
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(3, name="top-3-accuracy"),
    ]
    model.compile(optimizer=optimizer,
                loss=loss,
                metrics=metrics
                )

    callbacks = []
    callbacks.append(TensorboardCallback(log_dir=os.path.join(args.model_dir, "logs")))
    logger.info("*** Train ***")
    model.fit(
        train,
        validation_data=test,
        callbacks=callbacks,
        epochs=args.num_train_epochs,
    )
    metric = load_metric("rouge")

    results = evaluate(model, test)
    logger.info(results)
    model.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)