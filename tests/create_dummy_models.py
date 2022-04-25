import json
import os


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
from pathlib import Path
import collections.abc

from transformers.file_utils import is_tf_available, is_torch_available
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.processing_utils import ProcessorMixin, transformers_module




if not is_torch_available():
    raise ValueError("Please install PyTorch.")

if not is_tf_available():
    raise ValueError("Please install TensorFlow.")

import copy
import importlib
import os
import tempfile
from collections import OrderedDict

import h5py
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from transformers import (
    CONFIG_MAPPING,
    FEATURE_EXTRACTOR_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
    MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
    MODEL_FOR_OBJECT_DETECTION_MAPPING,
    MODEL_FOR_PRETRAINING_MAPPING,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    MODEL_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    PROCESSOR_MAPPING,
    TF_MODEL_FOR_CAUSAL_LM_MAPPING,
    TF_MODEL_FOR_MASKED_LM_MAPPING,
    TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
    TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
    TF_MODEL_FOR_PRETRAINING_MAPPING,
    TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    TF_MODEL_MAPPING,
    TF_MODEL_WITH_LM_HEAD_MAPPING,
    FLAX_MODEL_FOR_CAUSAL_LM_MAPPING,
    FLAX_MODEL_FOR_MASKED_LM_MAPPING,
    FLAX_MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
    FLAX_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
    FLAX_MODEL_FOR_PRETRAINING_MAPPING,
    FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    FLAX_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    FLAX_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    FLAX_MODEL_MAPPING,
    TOKENIZER_MAPPING,
    AutoFeatureExtractor,
    AutoTokenizer,
    logging,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.models.auto.configuration_auto import AutoConfig, model_type_to_module_name


INVALID_ARCH = []
logging.set_verbosity_error()

tokenizer_checkpoint_overrides = {"byt5": "google/byt5-small"}
ds = load_dataset("wikitext", "wikitext-2-raw-v1")
training_ds = ds["train"]
testing_ds = ds["test"]

per_model_type_configuration_attributes = {
    "big_bird": {"num_labels": 1},
}

unexportable_model_architectures = [
    "RoFormerForMultipleChoice",
    "TFRoFormerForMultipleChoice",
    "TFMobileBertForMultipleChoice",
    "MobileBertForMultipleChoice",
    "TFDistilBertForMultipleChoice",
    "DistilBertForMultipleChoice",
    "TFAlbertForMultipleChoice",
    "AlbertForMultipleChoice",
    "TFMPNetForMultipleChoice",
    "MPNetForMultipleChoice",
    "TFLongformerForMultipleChoice",
    "LongformerForMultipleChoice",
    "TFRobertaForMultipleChoice",
    "RobertaForMultipleChoice",
    "SqueezeBertForMultipleChoice",
    "TFSqueezeBertForMultipleChoice",
    "BertForMultipleChoice",
    "TFBertForMultipleChoice",
    "XLNetForMultipleChoice",
    "TFXLNetForMultipleChoice",
    "ElectraForMultipleChoice",
    "TFElectraForMultipleChoice",
    "FunnelForMultipleChoice",
    "TFFunnelForMultipleChoice",
]

# Define the PyTorch and TensorFlow mappings
pytorch_arch_mappings = [
    MODEL_MAPPING,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    MODEL_FOR_PRETRAINING_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
    MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
    MODEL_FOR_OBJECT_DETECTION_MAPPING,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
]

tensorflow_arch_mappings = [
    TF_MODEL_MAPPING,
    TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    TF_MODEL_FOR_MASKED_LM_MAPPING,
    TF_MODEL_FOR_PRETRAINING_MAPPING,
    TF_MODEL_FOR_CAUSAL_LM_MAPPING,
    TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
    TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
    TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    TF_MODEL_WITH_LM_HEAD_MAPPING,
    TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
]

flax_arch_mappings = [
    FLAX_MODEL_FOR_CAUSAL_LM_MAPPING,
    FLAX_MODEL_FOR_MASKED_LM_MAPPING,
    FLAX_MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
    FLAX_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
    FLAX_MODEL_FOR_PRETRAINING_MAPPING,
    FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    FLAX_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    FLAX_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    FLAX_MODEL_MAPPING,
]


def get_processor_types_from_config_class(config_class):

    processor_types = ()

    # Check first if a model has `ProcessorMixin`. If not, check `PreTrainedTokenizer` & `FeatureExtractionMixin`.
    if config_class in PROCESSOR_MAPPING:
        processor_types = PROCESSOR_MAPPING[config_class]
    elif config_class in TOKENIZER_MAPPING:
        processor_types = TOKENIZER_MAPPING[config_class]
    elif config_class in FEATURE_EXTRACTOR_MAPPING:
        processor_types = FEATURE_EXTRACTOR_MAPPING[config_class]
    else:
        # Some configurations have no processor at all. For example, generic composite models like
        # `EncoderDecoderModel` is used for any (compatible) text models. Also, `DecisionTransformer` doesn't
        # require any processor.
        # In these cases, we still add the configurations as keys but with `None` as value.
        pass

    # make a uniform format
    if not isinstance(processor_types, collections.abc.Sequence):
        processor_types = (processor_types,)

    # processor could be `None`. For example,
    # Keep only TODO:
    processor_types = tuple(p for p in processor_types if p is not None)

    return processor_types


def get_architectures_from_config_class(config_class, arch_mappings):
    """
    Map a configuration class to a tuple of all possible architectures attributed to that configuration.

    For example, BertConfig -> [BertModel, BertForMaskedLM, ..., BertForQuestionAnswering]
    """
    # A model architecture could appear in several mappings. For example, `BartForConditionalGeneration` is in
    #   - MODEL_FOR_PRETRAINING_MAPPING_NAMES
    #   - MODEL_WITH_LM_HEAD_MAPPING_NAMES
    #   - MODEL_FOR_MASKED_LM_MAPPING_NAMES
    #   - MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES
    # We avoid the duplication.
    architectures = set()

    for mapping in arch_mappings:
        if config_class in mapping:
            models = mapping[config_class]
            # TODO: Necessary?
            models = tuple(models) if isinstance(models, collections.abc.Sequence) else (models,)
            for model in models:
                # TODO: check the condition here
                if True or model.__name__ not in unexportable_model_architectures:
                    architectures.add(model)

    architectures = tuple(architectures)

    return architectures


def get_checkpoint_from_config_class(config_class):
    """
    """
    checkpoint = None

    import importlib
    import inspect
    import re

    # source code file where `config_class` is defined
    config_source_file = inspect.getsourcefile(config_class)

    # source code of `config_class`
    config_source = inspect.getsource(config_class)

    # module where `config_class` is defined.
    # (e.g. `transformers.models.bart.configuration_bert` for `BertConfig`)
    config_module = inspect.getmodule(config_class)
    config_module_name = config_module.__name__

    # get the model module
    model_module_name = config_module_name.replace("configuration_", "modeling_")
    # module where the corresponding model of `config_class` is defined
    # (e.g. `transformers.models.bart.modeling_bert` for `BertConfig`)
    try:
        model_module = importlib.import_module(model_module_name)
    except (ModuleNotFoundError, AttributeError):
        pass

    # regex used to find the checkpoint mentioned in the docstring of `config_class`.
    # For example, `[bert-base-uncased](https://huggingface.co/bert-base-uncased)`
    checkpoint_regex = re.compile("\[.+?\]\(https://huggingface\.co/.+?\)")
    checkpoints = checkpoint_regex.findall(config_source)

    # post processing
    for ckpt in checkpoints:

        regex = re.compile(r"(?:\[)(.+?)(?:\])")
        ckpt2 = regex.search(ckpt).group(1)
        ckpt_link = f"https://huggingface.co/{ckpt2}"
        if ckpt_link in ckpt:
            checkpoint = ckpt2
            break

    return checkpoint


# TODO: improve
def get_tiny_config(config_class):
    """
    Retrieve a tiny configuration from the configuration class. It uses each class' `ModelTester`.
    Args:
        configuration_class: Subclass of `PreTrainedConfig`.

    Returns:
        an instance of the configuration passed, with very small hyper-parameters

    """
    model_type = config_class.model_type
    camel_case_model_name = config_class.__name__.split("Config")[0]

    try:
        print("Importing", model_type_to_module_name(model_type))
        module_name = model_type_to_module_name(model_type)
        module = importlib.import_module(f".{module_name}.test_modeling_{module_name}", package="tests")
        model_tester_class = getattr(module, f"{camel_case_model_name}ModelTester", None)
    except ModuleNotFoundError:
        print(f"Will not build {model_type}: no model tester or cannot find the testing module from the model name.")
        return

    if model_tester_class is None:
        return

    model_tester = model_tester_class(parent=None)

    if hasattr(model_tester, "get_pipeline_config"):
        return model_tester.get_pipeline_config()
    elif hasattr(model_tester, "prepare_config_and_inputs"):
        # `PoolFormer` has no `get_config` defined. Furthermore, it's better to use `prepare_config_and_inputs` even if
        # `get_config` is defined, since there might be some extra change in `prepare_config_and_inputs`.
        return model_tester.prepare_config_and_inputs()[0]
    elif hasattr(model_tester, "get_config"):
        return model_tester.get_config()


def get_config_class_from_processor_class(processor_class):
    """
    Some configurations use other tokenizers/feature_extractors. For example, `GPT-J` use `GPT2Tokenizer`.
    If no checkpoint is found for a configuration, or a checkpoint is found without necessary file(s) to load the
    processor, it should be fine to use a checkpoint found for the config that corresponds to `processor_class`.
    """

    processor_prefix = processor_class.__name__
    for postfix in ["Processor", "TokenizerFast", "Tokenizer", "FeatureExtractor"]:
        processor_prefix = processor_prefix.replace(postfix, "")

    # TODO: bad Wav2Vec2CTCTokenizer without Wav2Vec2CTCConfig
    if processor_prefix == "Wav2Vec2CTC":
        processor_prefix = "Wav2Vec2"

    # Find the new configuration class
    new_config_name = f"{processor_prefix}Config"
    new_config_class = getattr(transformers_module, new_config_name)

    return new_config_class


def convert_tokenizer(tokenizer_fast: PreTrainedTokenizerFast):

    new_tokenizer = tokenizer_fast.train_new_from_iterator(training_ds["text"], 1024)
    new_tokenizer(testing_ds["text"])

    return new_tokenizer


def build_processor(config_class, processor_class):
    """Create and save a processor for `processor_class`.

    We don't save the processor here: the (same set of) processor will be saved in `build_model` for each `model_arch`,
    after some performed in `convert_processors`.
    """
    checkpoint = get_checkpoint_from_config_class(config_class)

    # TODO: checkpoint could be `None`. For example, `VisionTextDualEncoderConfig` has no checkpoint mentioned.
    # Try to get the checkpoint from `processor_class`
    if checkpoint is None:
        new_config_class = get_config_class_from_processor_class(processor_class)
        checkpoint = get_checkpoint_from_config_class(new_config_class)

    processor = None
    try:
        # TODO: Use Auto API
        processor = processor_class.from_pretrained(checkpoint)
    except Exception as e:
        # TODO
        pass

    if processor is None:

        # Try to build each component (tokenizer & feature extractor) of a `ProcessorMixin`.
        if issubclass(processor_class, ProcessorMixin):
            attrs = {}
            for attr_name in processor_class.attributes:
                attrs[attr_name] = []

                # This could be a tuple (for tokenizers). For example, `CLIPProcessor` has
                #   - feature_extractor_class = "CLIPFeatureExtractor"
                #   - tokenizer_class = ("CLIPTokenizer", "CLIPTokenizerFast")
                attr_class_names = getattr(processor_class, f"{attr_name}_class")
                if not isinstance(attr_class_names, tuple):
                    attr_class_names = (attr_class_names,)

                for name in attr_class_names:
                    attr_class = getattr(transformers_module, name)
                    attr = build_processor(config_class, attr_class)
                    if attr is not None:
                        attrs[attr_name].append(attr)

            # try to build a `ProcessorMixin`, so we can return the value
            if all(len(v) > 0 for v in attrs.values()):
                processor = processor_class(**{k: v[0] for k, v in attrs})

        else:
            # `checkpoint` might miss some files to load the processor. For example, `facebook/hubert-base-ls960`
            # has no tokenizer files to load `Wav2Vec2CTCTokenizer`.
            # Change `config_class` and call recursively.
            new_config_class = get_config_class_from_processor_class(processor_class)
            if new_config_class != config_class:
                processor = build_processor(new_config_class, processor_class)

    return processor


def convert_processors(processors):
    """Reduce the tokenizer's `vocab_size`, and update the slow tokenizer too (if any).

    Also remove the entries with `None` value.
    """

    tokenizers = []
    feature_extractors = []
    for processor in processors:
        if isinstance(processor, PreTrainedTokenizerBase):
            tokenizers.append(processor)
        elif isinstance(processor, FeatureExtractionMixin):
            feature_extractors.append(processor)
        elif isinstance(processor, ProcessorMixin):
            tokenizers.append(processor.tokenizer)
            tokenizers.append(processor.feature_extractor)

    fast_tokenizer = None
    slow_tokenizer = None
    for tokenizer in tokenizers:
        if isinstance(tokenizer, PreTrainedTokenizerFast):
            fast_tokenizer = tokenizer
            try:
                fast_tokenizer = convert_tokenizer(fast_tokenizer)
            except:
                pass
        else:
            slow_tokenizer = tokenizer

    if fast_tokenizer:
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                fast_tokenizer.save_pretrained(temp_dir, legacy_format=False)
                fast_tokenizer.save_pretrained(temp_dir, legacy_format=True)
                slow_tokenizer = AutoTokenizer.from_pretrained(temp_dir, fast_tokenizer=False)
            except:
                pass

    processors = [fast_tokenizer, slow_tokenizer] + feature_extractors
    processors = [p for p in processors if p is not None]

    return processors


def build_model(config_class, model_arch, output_folder, processors=None):
    """Create and save a model for `model_arch`.
    """

    # Get framework-agnostic architecture name. Used to save all PT/TF/Flax models into the same directory/repo.
    arch_name = model_arch.__name__
    if arch_name.startswith("TF"):
        arch_name = arch_name[2:]
    elif arch_name.startswith("Flax"):
        arch_name = arch_name[4:]

    output_folder = os.path.join(output_folder, arch_name)

    vocab_size = None
    # Save the (same set of) processors for each `model_arch` with the same `model_type`.
    for p in processors:
        p.save_pretrained(output_folder)
        if isinstance(p, PreTrainedTokenizerBase):
            vocab_size = p.vocab_size

    config_overrides = {"vocab_size": vocab_size}

    try:
        tiny_config = get_tiny_config(config_class)
        # TODO: `tiny_config` could be `None`. --> check and fix
        # For example, `VisionTextDualEncoderMixin` (despite it has `prepare_config_and_inputs` without impl.)
        if tiny_config is None:
            return None

        if config_overrides is not None:
            for k, v in config_overrides.items():
                setattr(tiny_config, k, v)

        model = model_arch(config=tiny_config)
        model.save_pretrained(output_folder)

    except Exception as e:
        return e

    return model


def build(config_class, to_create, output_folder):

    result = {k: {} for k in to_create}

    processor_classes = to_create["processor"]
    for processor_class in processor_classes:
        processor = build_processor(config_class, processor_class)
        result["processor"][processor_class] = processor

    # Try to reduce (fast) tokenizer's vocab size, and if successful, update the corresponding slow tokenizer (if any).
    processors = list(result["processor"].values())
    processors = convert_processors(processors)
    # update `result`
    result["processor"] = {type(p): p for p in processors}

    for pytorch_arch in to_create["pytorch"]:
        model = build_model(config_class, pytorch_arch, output_folder=output_folder, processors=processors)
        result["pytorch"][pytorch_arch] = model

    for tensorflow_arch in to_create["tensorflow"]:
        # Make PT/TF weights compatible
        pt_arch_name = tensorflow_arch.__name__[2:]  # Remove `TF`
        pt_arch = getattr(transformers_module, pt_arch_name)
        if isinstance(result["pytorch"].get(pt_arch, None), torch.nn.Module):
            ckpt = os.path.join(output_folder, pt_arch_name)
            # Use the same weights from PyTorch.
            try:
                model = tensorflow_arch.from_pretrained(ckpt, from_pt=True)
                model.save_pretrained(ckpt)
            except:
                # Conversion may fail. One example is, `FlaxWav2Vec2` doesn't support `config.do_stable_layer_norm=True`
                # yet.
                model = None
        else:
            model = build_model(config_class, tensorflow_arch, output_folder=output_folder, processors=processors)

        result["tensorflow"][tensorflow_arch] = model

    for flax_arch in to_create["flax"]:

        # Make PT/Flax weights compatible
        pt_arch_name = flax_arch.__name__[4:]  # Remove `Flax`
        pt_arch = getattr(transformers_module, pt_arch_name)
        if isinstance(result["pytorch"].get(pt_arch, None), torch.nn.Module):
            ckpt = os.path.join(output_folder, pt_arch_name)
            # Use the same weights from PyTorch.
            try:
                model = flax_arch.from_pretrained(ckpt, from_pt=True)
                model.save_pretrained(ckpt)
            except:
                # Conversion may fail. One example is, `FlaxWav2Vec2` doesn't support `config.do_stable_layer_norm=True`
                # yet.
                model = None
        else:
            model = build_model(config_class, flax_arch, output_folder=output_folder, processors=processors)

        result["flax"][flax_arch] = model

    return result


if __name__ == "__main__":

    def list_str(values):
        return values.split(",")

    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Will create all tiny models.")
    parser.add_argument(
        "--no_check",
        action="store_true",
        help="If set, will not check the validity of architectures. Use with caution.",
    )
    parser.add_argument(
        "-m",
        "--model_types",
        type=list_str,
        help="Comma-separated list of model type(s) from which the tiny models will be created.",
    )
    parser.add_argument("--black_list", type=list_str, help="Comma-separated list of model type(s) to ignore.",default='convbert,blenderbot-small,rag,dpr,retribert,layoutlmv2')
    ### parser.add_argument("output_path", type=Path, help="Path indicating where to store generated ONNX model.")
    args = parser.parse_args()

    # TEMP:
    args.output_path = "./temp/dummy/"
    args.all = True

    if not args.all and not args.model_types:
        raise ValueError("Please provide at least one model type or pass `--all` to export all architectures.")

    config_classes = CONFIG_MAPPING.values()
    if not args.all:
        config_classes = [CONFIG_MAPPING[model_type] for model_type in args.model_types]

    # Mappings from configs to processors/architectures
    processor_type_map = {c: get_processor_types_from_config_class(c) for c in config_classes}

    # Skip models that have no processor at all
    config_classes_with_processor = [c for c in config_classes if len(processor_type_map[c]) > 0]

    # Ignore some model types
    # TODO: Ask L
    if args.black_list:
        final_config_classes = [c for c in config_classes_with_processor if c.model_type not in args.black_list]

    to_create = {
        c: {
            "processor": processor_type_map[c],
            "pytorch": get_architectures_from_config_class(c, pytorch_arch_mappings),
            "tensorflow": get_architectures_from_config_class(c, tensorflow_arch_mappings),
            "flax": get_architectures_from_config_class(c, flax_arch_mappings),
        }
        for c in final_config_classes
    }

    # for c in final_config_classes:
    #     d = get_checkpoint_from_config_class(c)
    # exit(0)

    report = {"no_feature_extractor": [], "no_tokenizer": [], "identical_tokenizer": [], "vocab_sizes": {}}

    results = {}
    for c, _to_create in list(to_create.items())[:]:
        print(c)
        result = build(c, _to_create, output_folder=os.path.join(args.output_path, c.model_type))
        results[c] = result
        print("====================")

    _results = {}
    for k in results:
        #_results[str(k)] = {}
        for k1 in results[k]:
            #_results[str(k)][str(k1)] = {}
            for k2 in results[k][k1]:
                if isinstance(results[k][k1][k2], Exception):
                    if str(k) not in _results:
                        _results[str(k)] = {}
                    if str(k1) not in _results[str(k)]:
                        _results[str(k)][str(k1)] = {}
                    _results[str(k)][str(k1)][str(k2)] = str(results[k][k1][k2])

    with open("build_failed", "w") as fp:
        json.dump(_results, fp, ensure_ascii=True, indent=4)


