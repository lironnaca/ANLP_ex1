import os
os.environ['HF_DATASETS_CACHE'] = 'hf_cache/datasets'
os.environ['HF_METRICS_CACHE'] = 'hf_cache/metrics'
os.environ['HF_MODULES_CACHE'] = 'hf_cache/modules'
os.environ['HF_DATASETS_DOWNLOADED_EVALUATE_PATH'] = 'hf_cache/datasets_downloaded_evaluate'
os.environ['TRANSFORMERS_CACHE'] = 'transformers_cache'
import sys

from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
    TrainingArguments,
    Trainer,
    EvalPrediction
)
from datasets import load_dataset
from evaluate import load
import numpy as np


MODELS = {"bert-base-uncased": 512, "roberta-base": 512, "google/electra-base-generator": 512}
MODELS_BY_INDEX = {0: "bert-base-uncased", 1: "roberta-base", 2: "google/electra-base-generator"}
DATA_DIVISION = {"train": 67349, "validation": 872, "test": 1821}



def load_dataset_param(dataset_name):
    return load_dataset(dataset_name)


def load_model_and_tokenizer(model):
    config = AutoConfig.from_pretrained(model)
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForSequenceClassification.from_pretrained(model, config=config)
    return config, tokenizer, model


def tokenize_dataset(tokenizer, raw_dataset, max_length):
    def preprocess_function(examples):
        result = tokenizer(examples["sentence"], max_length=max_length, truncation=True)
        return result

    raw_dataset = raw_dataset.map(preprocess_function, batched=True)
    return raw_dataset


def data_according_to_user(raw_data, num, max_num):
    if num == -1:
        return raw_data
    return raw_data.select(range(min(num, max_num)))


def main(argv):
    data_training_arguments = {"number_of_seeds": int(argv[0]),
                               "number_of_samples_train": int(argv[1]),
                               "number_of_samples_validation": int(argv[2]),
                               "number_of_samples_test": int(argv[3])}
    train_time = 0

    raw_dataset = load_dataset("sst2")
    metric = load("accuracy")

    models_results = {}
    models_mean_results = np.array([])

    for model_name, max_length in MODELS.items():
        config, tokenizer, model = load_model_and_tokenizer(model_name)
        tokenized_dataset = tokenize_dataset(tokenizer, raw_dataset, max_length)

        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.argmax(preds, axis=1)
            result = metric.compute(predictions=preds, references=p.label_ids)
            return result

        results = []
        trainers = []

        for seed in range(data_training_arguments["number_of_seeds"]):
            set_seed(seed)

            train_dataset = data_according_to_user(tokenized_dataset["train"],
                                                   data_training_arguments["number_of_samples_train"], DATA_DIVISION["train"])
            validation_dataset = data_according_to_user(tokenized_dataset["validation"],
                                                        data_training_arguments["number_of_samples_validation"], DATA_DIVISION["validation"])

            trainer = Trainer(
                model=model,
                args= TrainingArguments("check",
                                        save_strategy="no",
                                        per_device_eval_batch_size=1),
                train_dataset=train_dataset,
                eval_dataset=validation_dataset,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer
            )

            train_result = trainer.train()
            train_time += train_result[2]["train_runtime"]

            results.append(trainer.evaluate()["eval_accuracy"])
            trainers.append(trainer)


        max_result = None
        max_result_index = None
        for i,result in enumerate(results):
          if max_result is None:
            max_result = result
            max_result_index = i
          elif result > max_result:
            max_result = result
            max_result_index = i

        models_results[model_name] = ([np.mean(results), np.std(results), max_result_index + 1, trainers[max_result_index]])
        models_mean_results = np.append(models_mean_results, np.mean(results))

    best_model = MODELS_BY_INDEX[np.argmax(models_mean_results)]
    best_seed_of_best_model = models_results[best_model][2]
    best_trained_model = models_results[best_model][3]

    set_seed(best_seed_of_best_model)

    test_dataset = data_according_to_user(tokenized_dataset["test"], data_training_arguments["number_of_samples_test"], DATA_DIVISION["test"])


    best_trained_model.model.eval()
    preds = best_trained_model.predict(test_dataset.remove_columns("label"))
    precitions = np.argmax(preds.predictions, axis=1)
    with open("res.txt", "w") as result:
        result.write(f"{MODELS_BY_INDEX[0]},{models_results[MODELS_BY_INDEX[0]][0]} +- {models_results[MODELS_BY_INDEX[0]][1]} \n")
        result.write(f"{MODELS_BY_INDEX[1]},{models_results[MODELS_BY_INDEX[1]][0]} +- {models_results[MODELS_BY_INDEX[1]][1]} \n")
        result.write(f"{MODELS_BY_INDEX[2]},{models_results[MODELS_BY_INDEX[2]][0]} +- {models_results[MODELS_BY_INDEX[2]][1]} \n")
        result.write("----\n")
        result.write(f"train time,{train_time}\n")
        result.write(f"predict time,{preds[2]['test_runtime']}\n")
    with open("predictions.txt", "w") as predictions_file:
        for i, input in enumerate(test_dataset):
            predictions_file.write(f"{input['sentence']}" + "###" + f"{precitions[i]}\n")


if __name__ == '__main__':
    main(sys.argv[1:])



