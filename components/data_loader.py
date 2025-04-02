import os
import re
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization

class DataLoader:
    def __init__(self, config):
        """
        Initialize the DataLoader with the provided configuration.
        """
        self.config = config
        self.vectorization = None

    def load_data(self, file_path):
        """
        Load data from a JSON file and return a DataFrame.
        """
        with open(file_path) as file:
            data = json.load(file)

        paths, keywords, captions = [], [], []
        for item in data:
            path = list(item.keys())[0]
            paths.append(path)
            keywords.append(list(item.values())[0]['keywords'])
            captions.append(list(item.values())[0]['clinical-description'])

        return pd.DataFrame({
            "image_path": paths,
            "keywords": keywords,
            "caption": captions
        })

    def preprocess_data(self, data):
        """
        Preprocess the data and return a dictionary of image paths to captions and keywords.
        """
        caption_mapping = {}
        text_data = []
        images_to_skip = set()

        for i in range(len(data)):
            img_name = os.path.join(self.config.data_dir, data['image_path'][i])
            caption = data['caption'][i].rstrip("\n")
            keywords = '[sep]'.join([kw.strip() for kw in data['keywords'][i].split(",")])

            # Filter captions based on sequence length
            tokens = caption.strip().split()
            if len(tokens) < self.config.min_seq_length or len(tokens) > self.config.max_seq_length:
                images_to_skip.add(img_name)
                continue

            if img_name.endswith("jpg") and img_name not in images_to_skip:
                caption = caption.replace('-', ' ')
                caption = "<start> " + caption.strip() + " <end>"
                text_data.append(caption)
                text_data.append(keywords)

                if img_name not in caption_mapping:
                    caption_mapping[img_name] = [caption]
                    caption_mapping[img_name].append(keywords)

        for img_name in images_to_skip:
            if img_name in caption_mapping:
                del caption_mapping[img_name]

        return caption_mapping, text_data

    def custom_standardization(self, input_string):
        """
        Standardize the input string by converting to lowercase and removing special characters.
        """
        lowercase = tf.strings.lower(input_string)
        return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(self.config.strip_chars), "")

    def build_vectorization(self, text_data):
        """
        Build and adapt the TextVectorization layer.
        """
        vectorization = TextVectorization(
            max_tokens=self.config.vocab_size,
            output_mode="int",
            output_sequence_length=self.config.max_seq_length,
            standardize=self.custom_standardization,
        )
        vectorization.adapt(text_data)
        return vectorization

    def decode_and_resize(self, img_path):
        """
        Decode and resize an image.
        """
        img = tf.io.read_file(img_path)
        img = tf.image.decode_image(img, channels=3)
        img.set_shape([self.config.image_size[0], self.config.image_size[1], 3])
        img = tf.image.resize(img, self.config.image_size)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

    def process_input(self, img_path, caption):
        """
        Process input image and caption for the model.
        """
        img = self.decode_and_resize(img_path)
        caption_vector = self.vectorization(caption)
        return (img, caption_vector[1]), caption_vector[0]

    def make_dataset(self, images, captions):
        """
        Create a TensorFlow dataset from images and captions.
        """
        dataset = tf.data.Dataset.from_tensor_slices((images, captions))
        dataset = dataset.shuffle(len(images))
        dataset = dataset.map(self.process_input, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    def load_datasets(self):
        """
        Load and preprocess the datasets, and return train, validation, and test datasets.
        """
        # Load data
        train_data = self.load_data(self.config.train_file)
        valid_data = self.load_data(self.config.val_file)
        test_data = self.load_data(self.config.test_file)

        # Preprocess data
        train_captions_mapping, train_text_data = self.preprocess_data(train_data)
        valid_captions_mapping, valid_text_data = self.preprocess_data(valid_data)
        test_captions_mapping, test_text_data = self.preprocess_data(test_data)

        # Combine training and validation data
        train_captions_mapping.update(valid_captions_mapping)
        train_text_data = np.concatenate((train_text_data, valid_text_data), axis=None)

        # Save the vocabulary to a file
        with open(self.config.corpus_file, "w") as f:
            for word in train_text_data:
                f.write(word + "\n")
        
        print(f"Corpus saved to {self.config.corpus_file}")

        # Build vectorization
        self.vectorization = self.build_vectorization(train_text_data)

        # Create datasets
        train_dataset = self.make_dataset(list(train_captions_mapping.keys()), list(train_captions_mapping.values()))
        valid_dataset = self.make_dataset(list(valid_captions_mapping.keys()), list(valid_captions_mapping.values()))
        test_dataset = self.make_dataset(list(test_captions_mapping.keys()), list(test_captions_mapping.values()))

        return train_dataset, valid_dataset, test_dataset, self.vectorization