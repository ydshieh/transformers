# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import shutil
import tempfile
import unittest

from transformers.models.seamless_m4t import SeamlessM4TFeatureExtractor
from transformers.models.wav2vec2 import Wav2Vec2CTCTokenizer
from transformers.models.wav2vec2.tokenization_wav2vec2 import VOCAB_FILES_NAMES
from transformers.models.wav2vec2_bert import Wav2Vec2BertProcessor
from transformers.utils import FEATURE_EXTRACTOR_NAME

from ...test_processing_common import ProcessorTesterMixin
from ..wav2vec2.test_feature_extraction_wav2vec2 import floats_list


class Wav2Vec2BertProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Wav2Vec2BertProcessor
    text_input_name = "labels"

    @classmethod
    def setUpClass(cls):
        vocab = "<pad> <s> </s> <unk> | E T A O N I H S R D L U M W C F G Y P B V K ' X J Q Z".split(" ")
        vocab_tokens = dict(zip(vocab, range(len(vocab))))

        cls.add_kwargs_tokens_map = {
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "bos_token": "<s>",
            "eos_token": "</s>",
        }
        feature_extractor_map = {
            "feature_size": 80,
            "padding_value": 0.0,
            "sampling_rate": 16000,
            "return_attention_mask": False,
            "do_normalize": True,
        }

        cls.tmpdirname = tempfile.mkdtemp()
        cls.vocab_file = os.path.join(cls.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        cls.feature_extraction_file = os.path.join(cls.tmpdirname, FEATURE_EXTRACTOR_NAME)
        with open(cls.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")

        with open(cls.feature_extraction_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(feature_extractor_map) + "\n")

        tokenizer = cls.get_tokenizer()
        tokenizer.save_pretrained(cls.tmpdirname)

    @classmethod
    def get_tokenizer(cls, **kwargs_init):
        kwargs = cls.add_kwargs_tokens_map.copy()
        kwargs.update(kwargs_init)
        return Wav2Vec2CTCTokenizer.from_pretrained(cls.tmpdirname, **kwargs)

    def get_feature_extractor(self, **kwargs):
        return SeamlessM4TFeatureExtractor.from_pretrained(self.tmpdirname, **kwargs)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    def test_save_load_pretrained_default(self):
        tokenizer = self.get_tokenizer()
        feature_extractor = self.get_feature_extractor()

        processor = Wav2Vec2BertProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        with tempfile.TemporaryDirectory() as tmpdir:
            processor.save_pretrained(tmpdir)
            processor = Wav2Vec2BertProcessor.from_pretrained(tmpdir)

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertIsInstance(processor.tokenizer, Wav2Vec2CTCTokenizer)

        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor.to_json_string())
        self.assertIsInstance(processor.feature_extractor, SeamlessM4TFeatureExtractor)

    def test_save_load_pretrained_additional_features(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = Wav2Vec2BertProcessor(
                tokenizer=self.get_tokenizer(), feature_extractor=self.get_feature_extractor()
            )
            processor.save_pretrained(tmpdir)

            tokenizer_add_kwargs = Wav2Vec2CTCTokenizer.from_pretrained(
                tmpdir, **(self.add_kwargs_tokens_map | {"bos_token": "(BOS)", "eos_token": "(EOS)"})
            )
            feature_extractor_add_kwargs = SeamlessM4TFeatureExtractor.from_pretrained(
                tmpdir, do_normalize=False, padding_value=1.0
            )

            processor = Wav2Vec2BertProcessor.from_pretrained(
                tmpdir, bos_token="(BOS)", eos_token="(EOS)", do_normalize=False, padding_value=1.0
            )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.tokenizer, Wav2Vec2CTCTokenizer)

        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.feature_extractor, SeamlessM4TFeatureExtractor)

    def test_feature_extractor(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = Wav2Vec2BertProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        raw_speech = floats_list((3, 1000))

        input_feat_extract = feature_extractor(raw_speech, return_tensors="np")
        input_processor = processor(raw_speech, return_tensors="np")

        for key in input_feat_extract:
            self.assertAlmostEqual(input_feat_extract[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_tokenizer(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = Wav2Vec2BertProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        input_str = "This is a test string"
        encoded_processor = processor(text=input_str)

        encoded_tok = tokenizer(input_str)

        for key in encoded_tok:
            self.assertListEqual(encoded_tok[key], encoded_processor[key])

    def test_tokenizer_decode(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = Wav2Vec2BertProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        predicted_ids = [[1, 4, 5, 8, 1, 0, 8], [3, 4, 3, 1, 1, 8, 9]]

        decoded_processor = processor.batch_decode(predicted_ids)
        decoded_tok = tokenizer.batch_decode(predicted_ids)

        self.assertListEqual(decoded_tok, decoded_processor)

    def test_model_input_names(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = Wav2Vec2BertProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        self.assertListEqual(
            processor.model_input_names,
            feature_extractor.model_input_names,
            msg="`processor` and `feature_extractor` model input names do not match",
        )
