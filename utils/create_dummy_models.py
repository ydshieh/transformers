import json
import os
import shutil
import sys

sys.path.append(".")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
from pathlib import Path
import collections.abc

from transformers.file_utils import is_tf_available, is_torch_available
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.image_utils import ImageFeatureExtractionMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.processing_utils import ProcessorMixin, transformers_module
from check_config_docstrings import get_checkpoint_from_config_class


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
    # "RoFormerForMultipleChoice",
    # "TFRoFormerForMultipleChoice",
    # "TFMobileBertForMultipleChoice",
    # "MobileBertForMultipleChoice",
    # "TFDistilBertForMultipleChoice",
    # "DistilBertForMultipleChoice",
    # "TFAlbertForMultipleChoice",
    # "AlbertForMultipleChoice",
    # "TFMPNetForMultipleChoice",
    # "MPNetForMultipleChoice",
    # "TFLongformerForMultipleChoice",
    # "LongformerForMultipleChoice",
    # "TFRobertaForMultipleChoice",
    # "RobertaForMultipleChoice",
    # "SqueezeBertForMultipleChoice",
    # "TFSqueezeBertForMultipleChoice",
    # "BertForMultipleChoice",
    # "TFBertForMultipleChoice",
    # "XLNetForMultipleChoice",
    # "TFXLNetForMultipleChoice",
    # "ElectraForMultipleChoice",
    # "TFElectraForMultipleChoice",
    # "FunnelForMultipleChoice",
    # "TFFunnelForMultipleChoice",
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
    """Return a tuple of processors for `config_class`

    We use `tuple` here to include (potentially) both slow & fast tokenizers.
    """

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
        pass

    # make a uniform return type
    if not isinstance(processor_types, collections.abc.Sequence):
        processor_types = (processor_types,)
    else:
        processor_types = tuple(processor_types)

    # We might get `None` for some tokenizers - remove them here.
    processor_types = tuple(p for p in processor_types if p is not None)

    return processor_types


def get_architectures_from_config_class(config_class, arch_mappings):
    """
    Get a tuple of all possible architectures attributed to a configuration class.

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
            models = tuple(models) if isinstance(models, collections.abc.Sequence) else (models,)
            for model in models:
                if model.__name__ not in unexportable_model_architectures:
                    architectures.add(model)

    architectures = tuple(architectures)

    return architectures


# TODO: improve
def get_tiny_config(config_class):
    """
    Retrieve a tiny configuration from the configuration class. It uses each class' `ModelTester`.
    Args:
        configuration_class: Subclass of `PreTrainedConfig`.

    Returns:
        An instance of `configuration_class` with tiny hyper-parameters
    """
    model_type = config_class.model_type
    camel_case_model_name = config_class.__name__.split("Config")[0]

    try:
        print("Importing", model_type_to_module_name(model_type))
        module_name = model_type_to_module_name(model_type)
        module = importlib.import_module(f".models.{module_name}.test_modeling_{module_name}", package="tests")
        model_tester_class = getattr(module, f"{camel_case_model_name}ModelTester", None)
    except ModuleNotFoundError as e:
        print(f"Tiny config not created for {model_type}: cannot find the testing module from the model name.")
        raise ValueError(f"Tiny config not created for {model_type} - cannot find the testing module from the model name: {str(e)}")

    if model_tester_class is None:
        print(f"Tiny config not created for {model_type}: no model tester is found in the testing module.")
        raise ValueError(f"Tiny config not created for {model_type}: no model tester is found in the testing module.")

    # `parent` is an instance of `unittest.TestCase`, but we don't need it.
    model_tester = model_tester_class(parent=None)

    if hasattr(model_tester, "get_pipeline_config"):
        return model_tester.get_pipeline_config()
    elif hasattr(model_tester, "prepare_config_and_inputs"):
        # `PoolFormer` has no `get_config` defined. Furthermore, it's better to use `prepare_config_and_inputs` even if
        # `get_config` is defined, since there might be some extra changes in `prepare_config_and_inputs`.
        return model_tester.prepare_config_and_inputs()[0]
    elif hasattr(model_tester, "get_config"):
        return model_tester.get_config()
    else:
        print(f"Tiny config not created for {model_type}: the model tester {model_tester_class.__name__} lacks necessary method to create config.")
        raise ValueError(f"Tiny config not created for {model_type}: the model tester {model_tester_class.__name__} lacks necessary method to create config.")


def get_config_class_from_processor_class(processor_class):
    """
    Some configurations use tokenizers/feature_extractors from other models. For example, `GPT-J` uses `GPT2Tokenizer`.
    If no checkpoint is found for a configuration, or a checkpoint is found without necessary file(s) to load the
    processor, we get the config class that corresponds to `processor_class` and use it to find a checkpoint in order to
    create the processor.
    """

    processor_prefix = processor_class.__name__
    for postfix in ["Processor", "TokenizerFast", "Tokenizer", "FeatureExtractor"]:
        processor_prefix = processor_prefix.replace(postfix, "")

    # `Wav2Vec2CTCTokenizer` -> `Wav2Vec2Config`
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
    """Create a processor for `processor_class`.

    The processor is not saved here. Instead, it will be saved in `convert_processors` after further changes in
    `convert_processors`. For each model architecture`, a copy will be created and saved along the model.
    """
    # Currently, this solely uses the docstring in the source file of `config_class` to find a checkpoint.
    checkpoint = get_checkpoint_from_config_class(config_class)

    if checkpoint is None:
        # try to get the checkpoint from the config class for `processor_class`
        config_class_from_processor_class = get_config_class_from_processor_class(processor_class)
        checkpoint = get_checkpoint_from_config_class(config_class_from_processor_class)

    processor = None
    try:
        # TODO: Maybe use Auto API
        processor = processor_class.from_pretrained(checkpoint)
    except Exception as e:
        pass

    # Try to get a new processor class from checkpoint. This is helpful to deal with a checkpoint without necessary file
    # to load processor while `processor_class` is an Auto class.
    # For example, see `https://huggingface.co/asapp/sew-tiny-100k`.
    if processor is None and checkpoint is not None and issubclass(processor_class, (PreTrainedTokenizerBase, AutoTokenizer)):
        config = AutoConfig.from_pretrained(checkpoint)
        assert isinstance(config, config_class)
        tokenizer_class = config.tokenizer_class
        if tokenizer_class is not None:
            new_processor_class = getattr(transformers_module, tokenizer_class)
            if new_processor_class != processor_class:
                processor = build_processor(config_class, new_processor_class)

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

            # try to build a `ProcessorMixin`, so we can return a value
            if all(len(v) > 0 for v in attrs.values()):
                try:
                    processor = processor_class(**{k: v[0] for k, v in attrs.items()})
                except Exception as e:
                    pass
        else:
            # `checkpoint` might lack some file(s) to load the processor. For example, `facebook/hubert-base-ls960`
            # has no tokenizer file to load `Wav2Vec2CTCTokenizer`.
            config_class_from_processor_class = get_config_class_from_processor_class(processor_class)
            if config_class_from_processor_class != config_class:
                processor = build_processor(config_class_from_processor_class, processor_class)

    # validation
    if processor is not None:
        assert isinstance(processor, processor_class) or processor_class.__name__.startswith("Auto")

    return processor


def convert_processors(processors, output_folder, result):
    """Reduce `vocab_size` in tokenizer(s)"""
    tokenizers = []
    feature_extractors = []
    for processor in processors:
        if isinstance(processor, PreTrainedTokenizerBase):
            tokenizers.append(processor)
        elif isinstance(processor, FeatureExtractionMixin):
            feature_extractors.append(processor)
        elif isinstance(processor, ProcessorMixin):
            # Currently, we only have these 2 possibilities
            tokenizers.append(processor.tokenizer)
            feature_extractors.append(processor.feature_extractor)

    # check the built processors have the unique type
    assert len(set([x.__class__.__name__ for x in feature_extractors])) < 2
    assert len(set([x.__class__.__name__.replace("Fast", "") for x in tokenizers])) < 2

    fast_tokenizer = None
    slow_tokenizer = None
    for tokenizer in tokenizers:
        if isinstance(tokenizer, PreTrainedTokenizerFast):
            if fast_tokenizer is None:
                fast_tokenizer = tokenizer
                try:
                    fast_tokenizer = convert_tokenizer(tokenizer)
                except Exception as e:
                    result["warnings"].append(f"Failed to convert the fast tokenizer for {fast_tokenizer.__class__.__name__}: {e}")
                    continue
        elif slow_tokenizer is None:
            slow_tokenizer = tokenizer

    if fast_tokenizer:
        slow_tokenizer = None
        try:
            fast_tokenizer.save_pretrained(output_folder)
        except Exception as e:
            result["warnings"].append(f"Failed to save the fast tokenizer for {fast_tokenizer.__class__.__name__}: {e}")
            fast_tokenizer = None

        if fast_tokenizer:
            try:
                slow_tokenizer = AutoTokenizer.from_pretrained(output_folder, use_fast=False)
            except Exception as e:
                result["warnings"].append(f"Failed to load the slow tokenizer saved from {fast_tokenizer.__class__.__name__}: {e}")
                pass

    elif slow_tokenizer:
        slow_tokenizer.save_pretrained(output_folder)

    processors = [fast_tokenizer, slow_tokenizer] + feature_extractors
    processors = [p for p in processors if p is not None]
    for p in processors:
        p.save_pretrained(output_folder)

    return processors


def build_model(config_class, model_arch, output_folder, config_overrides=None):
    """Create and save a model for `model_arch`.
    """
    # Get framework-agnostic architecture name. Used to save all PT/TF/Flax models into the same directory/repo.
    arch_name = model_arch.__name__
    if arch_name.startswith("TF"):
        arch_name = arch_name[2:]
    elif arch_name.startswith("Flax"):
        arch_name = arch_name[4:]

    processor_output_folder = os.path.join(output_folder, "processors")
    model_output_folder = os.path.join(output_folder, arch_name)
    # copy the (same set of) processors to the model specific folder
    if os.path.isdir(processor_output_folder):
        shutil.copytree(processor_output_folder, model_output_folder, dirs_exist_ok=True)

    tiny_config = get_tiny_config(config_class)

    if config_overrides is not None:
        for k, v in config_overrides.items():
            setattr(tiny_config, k, v)

    model = model_arch(config=tiny_config)
    model.save_pretrained(model_output_folder)
    model.from_pretrained(model_output_folder)

    return model


def build(config_class, to_create, output_folder):

    result = {k: {} for k in to_create}
    result["error"] = None
    result["warnings"] = []

    # build processors
    processor_classes = to_create["processor"]
    for processor_class in processor_classes:
        processor = build_processor(config_class, processor_class)
        if processor is not None:
            result["processor"][processor_class] = processor

    if len(result["processor"]) == 0:
        result["error"] = "No processor could be built."
        return result

    # Reduce the vocab size in tokenizer(s)
    processors = list(result["processor"].values())
    processor_output_folder = os.path.join(output_folder, "processors")
    processors = convert_processors(processors, processor_output_folder, result)
    # update `result`
    result["processor"] = {type(p): p for p in processors}

    if len(result["processor"]) == 0:
        result["error"] = "No processor could be converted."
        return result

    for processor in processors:
        if isinstance(processor, PreTrainedTokenizerBase):
            vocab_size = processor.vocab_size
            result["vocab_size"] = vocab_size
        elif isinstance(processor, ImageFeatureExtractionMixin):
            image_size = getattr(processor, "size", None)
            crop_size = getattr(processor, "crop_size", None)
            crop_pct = getattr(processor, "crop_pct", None)
            result["image_size"] = image_size
            result["crop_size"] = crop_size
            result["crop_pct"] = crop_pct

    config_overrides = {k: v for k, v in result.items() if k in ["vocab_size", "image_size"] and v is not None}

    for pytorch_arch in to_create["pytorch"]:
        result["pytorch"][pytorch_arch] = {}
        try:
            model = build_model(config_class, pytorch_arch, output_folder=output_folder, config_overrides=config_overrides)
        except Exception as e:
            model = None
            result["pytorch"][pytorch_arch]["error"] = f"Failed to build the model: {e}"

        result["pytorch"][pytorch_arch]["model"] = model

    # TODO: remove
    return result

    # TODO: continue

    # for tensorflow_arch in to_create["tensorflow"]:
    #     # Make PT/TF weights compatible
    #     pt_arch_name = tensorflow_arch.__name__[2:]  # Remove `TF`
    #     pt_arch = getattr(transformers_module, pt_arch_name)
    #     if isinstance(result["pytorch"].get(pt_arch, None), torch.nn.Module):
    #         ckpt = os.path.join(output_folder, pt_arch_name)
    #         # Use the same weights from PyTorch.
    #         try:
    #             model = tensorflow_arch.from_pretrained(ckpt, from_pt=True)
    #             model.save_pretrained(ckpt)
    #         except:
    #             # Conversion may fail. One example is, `FlaxWav2Vec2` doesn't support `config.do_stable_layer_norm=True`
    #             # yet.
    #             model = None
    #     else:
    #         model = build_model(config_class, tensorflow_arch, output_folder=output_folder, processors=processors)
    #
    #     result["tensorflow"][tensorflow_arch] = model

    # for flax_arch in to_create["flax"]:
    #
    #     # Make PT/Flax weights compatible
    #     pt_arch_name = flax_arch.__name__[4:]  # Remove `Flax`
    #     pt_arch = getattr(transformers_module, pt_arch_name)
    #     if isinstance(result["pytorch"].get(pt_arch, None), torch.nn.Module):
    #         ckpt = os.path.join(output_folder, pt_arch_name)
    #         # Use the same weights from PyTorch.
    #         try:
    #             model = flax_arch.from_pretrained(ckpt, from_pt=True)
    #             model.save_pretrained(ckpt)
    #         except:
    #             # Conversion may fail. One example is, `FlaxWav2Vec2` doesn't support `config.do_stable_layer_norm=True`
    #             # yet.
    #             model = None
    #     else:
    #         model = build_model(config_class, flax_arch, output_folder=output_folder, processors=processors)
    #
    #     result["flax"][flax_arch] = model

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
    parser.add_argument("--black_list", type=list_str, help="Comma-separated list of model type(s) to ignore.", default='convbert,blenderbot-small,rag,dpr,retribert,layoutlmv2')
    # TODO: (remove) removed `ONNX` from the original `help`.
    # TODO: (remove) remove #
    # parser.add_argument("output_path", type=Path, help="Path indicating where to store generated model.")
    args = parser.parse_args()

    # --------------------------------------------------------------------------------
    # TODO: (remove)
    args.output_path = "./temp/dummy/"
    args.all = True
    # --------------------------------------------------------------------------------

    if not args.all and not args.model_types:
        raise ValueError("Please provide at least one model type or pass `--all` to export all architectures.")

    config_classes = CONFIG_MAPPING.values()
    if not args.all:
        config_classes = [CONFIG_MAPPING[model_type] for model_type in args.model_types]

    # TODO: (remove) changed `get_X_from_configuration_list` to `get_X_from_config_class`
    # Mappings from config classes to lists of processor (tokenizer, feature extractor, processor) classes
    processor_type_map = {c: get_processor_types_from_config_class(c) for c in config_classes}

    # Skip models that have no processor at all
    config_classes_with_processor = [c for c in config_classes if len(processor_type_map[c]) > 0]

    # Ignore some model types
    # TODO: Discuss with Lysandre about the reason
    if args.black_list:
        final_config_classes = [c for c in config_classes_with_processor if c.model_type not in args.black_list]

    to_create = {
        c: {
            "processor": processor_type_map[c],
            "pytorch": get_architectures_from_config_class(c, pytorch_arch_mappings),
            "tensorflow": get_architectures_from_config_class(c, tensorflow_arch_mappings),
            #"flax": get_architectures_from_config_class(c, flax_arch_mappings),
        }
        for c in final_config_classes
    }

    # TODO: (to be continued)

    results = {}
    for c, _to_create in list(to_create.items())[:]:
        print(c)
        result = build(c, _to_create, output_folder=os.path.join(args.output_path, c.model_type))
        results[c] = result
        print("====================")

    # serialization
    for c, result in results.items():
        for framework in ["pytorch", "tensorflow", "flax"]:
            if framework in result:
                for model_arch in result[framework]:
                    if result[framework][model_arch]["model"] is not None:
                        result[framework][model_arch]["model"] = result[framework][model_arch]["model"].__class__.__name__

    _results = copy.deepcopy(results)
    for c, result in results.items():
        _results[c.__name__] = _results[c]
        del _results[c]
        for k in result["processor"]:
            _results[c.__name__]["processor"][k.__name__] = str(result["processor"][k])
            del _results[c.__name__]["processor"][k]
        for framework in ["pytorch", "tensorflow", "flax"]:
            if framework in result:
                for model_arch in result[framework]:
                    _results[c.__name__][framework][model_arch.__name__] = _results[c.__name__][framework][model_arch]
                    del _results[c.__name__][framework][model_arch]

    # TODO: remove
    with open("dummy_creation.json", "w") as fp:
        json.dump(_results, fp, indent=4)

    # TODO: remove
    exit(0)

    report = {"no_feature_extractor": [], "no_tokenizer": [], "identical_tokenizer": [], "vocab_sizes": {}}

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

    print("--- Report ---")

    config_classes_without_processor = [c for c in config_classes if len(processor_type_map[c]) == 0]
    if len(config_classes_without_processor) > 0:
        print(
            f"Some models could not be exported due to a lack of processor: {[c.model_type for c in config_classes_without_processor]}"
        )
