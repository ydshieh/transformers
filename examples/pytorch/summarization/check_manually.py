import numpy as np
import torch

from transformers import AutoTokenizer
from transformers import LEDModel, LEDForConditionalGeneration

import datasets

summarization_name_mapping = {
    "cnn_dailymail": ("article", "highlights"),
    "xsum": ("document", "summary"),
}

ckpt_led_base = "allenai/led-base-16384"
ckpt_led_large = "allenai/led-large-16384"

tokenizer = AutoTokenizer.from_pretrained(ckpt_led_base)
model = LEDForConditionalGeneration.from_pretrained(ckpt_led_base)

def get_dataset(dataset_name):

    max_source_length = 1024
    max_target_length = 128
    padding = True
    ignore_pad_token_for_loss = True
    padding = "max_length"
    prefix = ""
    max_train_samples = 1024
    max_eval_samples = 256
    preprocessing_num_workers = 8

    raw_datasets = datasets.load_dataset(dataset_name)

    text_column, summary_column = summarization_name_mapping[dataset_name]

    def foo(x):

        if x == tokenizer.cls_token_id:
            return 1
        elif x == tokenizer.pad_token_id:
            return -1
        else:
            return 0

    def preprocess_function(examples):
        # remove pairs where at least one record is None

        inputs, targets = [], []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] and examples[summary_column][i]:
                inputs.append(examples[text_column][i])
                targets.append(examples[summary_column][i])

        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]

        if model.__class__.__name__.startswith("LED"):
            model_inputs["global_attention_mask"] = [[foo(y) for y in x] for x in model_inputs["input_ids"]]

        decoder_input_ids = model.prepare_decoder_input_ids_from_labels(labels=torch.tensor(model_inputs["labels"], dtype=torch.int32))
        decoder_input_ids = decoder_input_ids.numpy().tolist()
        model_inputs["decoder_input_ids"] = decoder_input_ids

        return model_inputs

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["validation"]

    train_dataset = train_dataset.select(range(max_train_samples))
    eval_dataset = eval_dataset.select(range(max_eval_samples))

    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=preprocessing_num_workers,
        remove_columns=['document', 'summary', 'id'],
        desc="Running tokenizer on train dataset",
    )
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=preprocessing_num_workers,
        remove_columns=['document', 'summary', 'id'],
        desc="Running tokenizer on validation dataset",
    )

    return train_dataset, eval_dataset

train_dataset, eval_dataset = get_dataset("xsum")
for idx, eval_example in enumerate(eval_dataset):

    eval_example.pop("labels")

    decoder_input_ids = eval_example.pop("decoder_input_ids")
    eval_example["decoder_input_ids"] = [2, 0] + decoder_input_ids[2:5]

    for k in eval_example:
        eval_example[k] = torch.tensor([eval_example[k]], dtype=torch.int32)

    model.led.decoder.buffer = {}
    output = model(**eval_example)

    print(f"example idx: {idx}")

    for k in model.led.decoder.buffer:
        h = model.led.decoder.buffer[k]
        if not isinstance(h, dict):
            pass
            # print(f'max diff in {k}: {np.amax(np.abs((h[0, 0] - h[0, 1]).detach().to("cpu").numpy()))}')
        else:
            layer_idx = k
            buffer = h
            for name in buffer:
                h = buffer[name]
                #print(f'layer {layer_idx} - {name}: max <eos> = {torch.max(torch.abs(h[0, 0]))}')
                #print(f'layer {layer_idx} - {name}: max <bos> = {torch.max(torch.abs(h[0, 1]))}')
                #print(f'layer {layer_idx} - {name}: max <eos> dim = {torch.argmax(torch.abs(h[0, 0]), dim=-1)}')
                #print(f'layer {layer_idx} - {name}: max <bos> dim = {torch.argmax(torch.abs(h[0, 1]), dim=-1)}')
                #top = torch.topk(torch.abs(h[0, 0]), k=8, dim=-1, largest=True, sorted=True)
                #print(f'layer {layer_idx} - {name}: top <eos> indices = {top.indices}')
                #print(f'layer {layer_idx} - {name}: top <eos> values = {top.values}')
                #print(f'layer {layer_idx} - {name}: var <eos> = {torch.var(h[0, 0], unbiased=False)}')
                #print(f'layer {layer_idx} - {name}: var <bos> = {torch.var(h[0, 1], unbiased=False)}')
                if "hidden_states: ffn: final_layer_norm" in name:
                    print(f'max diff in layer {layer_idx} - {name}: {np.amax(np.abs((h[0, 0] - h[0, 1]).detach().to("cpu").numpy()))}')
                    print(f"-" * 20)

    print(f'max diff in lm logits: {np.amax(np.abs((output.logits[0, 0] - output.logits[0, 1]).detach().to("cpu").numpy()))}')
    print(f"-" * 20)

    pred = torch.argmax(output.logits, dim=-1).detach().to("cpu").numpy().tolist()
    print(f'predidcted token ids: {pred}')

    print(f"=" * 40)

    if idx >= 10:
        break
