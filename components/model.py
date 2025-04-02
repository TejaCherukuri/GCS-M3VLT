import keras
from keras import layers
from keras import backend as K
from keras.layers import Layer
from keras.layers import Conv1D, Conv2D, Conv3D, Reshape, Activation, Softmax, Permute, add, dot

import numpy as np

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0

from tqdm import tqdm
from rouge import Rouge
from nltk.translate.bleu_score import corpus_bleu

# Reference: https://github.com/titu1994/keras-global-context-networks/blob/master/gc.py
def GlobalContextBlock(ip, reduction_ratio=8, transform_activation='linear'):

    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
    ip_shape = K.int_shape(ip)

    # check rank and calculate the input shape
    if len(ip_shape) == 3:  # temporal / time series data
        rank = 3
        batchsize, dim1, channels = ip_shape

    elif len(ip_shape) == 4:  # spatial / image data
        rank = 4

        if channel_dim == 1:
            batchsize, channels, dim1, dim2 = ip_shape
        else:
            batchsize, dim1, dim2, channels = ip_shape

    elif len(ip_shape) == 5:  # spatio-temporal / Video or Voxel data
        rank = 5

        if channel_dim == 1:
            batchsize, channels, dim1, dim2, dim3 = ip_shape
        else:
            batchsize, dim1, dim2, dim3, channels = ip_shape

    else:
        raise ValueError('Input dimension has to be either 3 (temporal), 4 (spatial) or 5 (spatio-temporal)')

    if rank > 3:
        flat_spatial_dim = -1 if K.image_data_format() == 'channels_first' else 1
    else:
        flat_spatial_dim = 1

    """ Context Modelling Block """
    # [B, ***, C] or [B, C, ***]
    input_flat = _spatial_flattenND(ip, rank)
    # [B, ..., C] or [B, C, ...]
    context = _convND(ip, rank, channels=1, kernel=1)
    # [B, ..., 1] or [B, 1, ...]
    context = _spatial_flattenND(context, rank)
    # [B, ***, 1] or [B, 1, ***]
    context = Softmax(axis=flat_spatial_dim)(context)

    # Compute context block outputs
    context = dot([input_flat, context], axes=flat_spatial_dim)
    # [B, C, 1]
    context = _spatial_expandND(context, rank)
    # [B, C, 1...] or [B, 1..., C]

    """ Transform block """
    # Transform bottleneck
    # [B, C // R, 1...] or [B, 1..., C // R]
    transform = _convND(context, rank, channels // reduction_ratio, kernel=1)
    # Group normalization acts as Layer Normalization when groups = 1
    #transform = GroupNormalization(groups=1, axis=channel_dim)(transform)
    transform = Activation('relu')(transform)

    # Transform output block
    # [B, C, 1...] or [B, 1..., C]
    transform = _convND(transform, rank, channels, kernel=1)
    transform = Activation(transform_activation)(transform)

    # apply context transform
    out = add([ip, transform])

    return out


def _convND(ip, rank, channels, kernel=1):
    assert rank in [3, 4, 5], "Rank of input must be 3, 4 or 5"

    if rank == 3:
        x = Conv1D(channels, kernel, padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
    elif rank == 4:
        x = Conv2D(channels, (kernel, kernel), padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
    else:
        x = Conv3D(channels, (kernel, kernel, kernel), padding='same', use_bias=False, kernel_initializer='he_normal')(ip)

    return x


def _spatial_flattenND(ip, rank):
    assert rank in [3, 4, 5], "Rank of input must be 3, 4 or 5"

    ip_shape = K.int_shape(ip)
    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1

    if rank == 3:
        x = ip  # identity op for rank 3

    elif rank == 4:
        if channel_dim == 1:
            # [C, D1, D2] -> [C, D1 * D2]
            shape = [ip_shape[1], ip_shape[2] * ip_shape[3]]
        else:
            # [D1, D2, C] -> [D1 * D2, C]
            shape = [ip_shape[1] * ip_shape[2], ip_shape[3]]

        x = Reshape(shape)(ip)

    else:
        if channel_dim == 1:
            # [C, D1, D2, D3] -> [C, D1 * D2 * D3]
            shape = [ip_shape[1], ip_shape[2] * ip_shape[3] * ip_shape[4]]
        else:
            # [D1, D2, D3, C] -> [D1 * D2 * D3, C]
            shape = [ip_shape[1] * ip_shape[2] * ip_shape[3], ip_shape[4]]

        x = Reshape(shape)(ip)

    return x


def _spatial_expandND(ip, rank):
    assert rank in [3, 4, 5], "Rank of input must be 3, 4 or 5"

    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1

    if rank == 3:
        x = Permute((2, 1))(ip)  # identity op for rank 3

    elif rank == 4:
        if channel_dim == 1:
            # [C, D1, D2] -> [C, D1 * D2]
            shape = [-1, 1, 1]
        else:
            # [D1, D2, C] -> [D1 * D2, C]
            shape = [1, 1, -1]

        x = Reshape(shape)(ip)

    else:
        if channel_dim == 1:
            # [C, D1, D2, D3] -> [C, D1 * D2 * D3]
            shape = [-1, 1, 1, 1]
        else:
            # [D1, D2, D3, C] -> [D1 * D2 * D3, C]
            shape = [1, 1, 1, -1]

        x = Reshape(shape)(ip)

    return x

class AttentionGate(Layer):
    def __init__(self, filters, **kwargs):
        self.filters = filters
        super(AttentionGate, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create trainable parameters for attention gate
        self.Wx = self.add_weight(name='Wx',
                                 shape=(input_shape[0][-1], self.filters),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.Wg = self.add_weight(name='Wg',
                                 shape=(input_shape[1][-1], self.filters),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.psi = self.add_weight(name='psi',
                                  shape=(self.filters, 1),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.bxg = self.add_weight(name='bxg',
                                  shape=(self.filters,),
                                  initializer='zeros',
                                  trainable=True)
        self.bpsi = self.add_weight(name='bpsi',
                                   shape=(1,),
                                   initializer='zeros',
                                   trainable=True)
        super(AttentionGate, self).build(input_shape)

    def call(self, inputs):
        xl, g = inputs

        # Compute additive attention
        att = K.relu(K.dot(xl, self.Wx) + K.dot(g, self.Wg) + self.bxg)
        att = K.dot(att, self.psi) + self.bpsi
        att = K.sigmoid(att)

        # Apply attention gate
        x_hat = att * xl

        return x_hat

    def compute_output_shape(self, input_shape):
        return input_shape[0]


def CNNEncoder(config):
    base_model = EfficientNetV2B0(
        input_shape=(*config.image_size, 3), include_top=False, weights="imagenet",
    )
    
    base_model.trainable = True
    
    fmaps = base_model.output
    
    context_fmaps = GlobalContextBlock(fmaps) 
    att_fmaps = AttentionGate(fmaps.shape[-1])([fmaps, context_fmaps])
    
    model_out = layers.Reshape((-1, att_fmaps.shape[-1]))(att_fmaps)
    
    image_encoder = keras.models.Model(base_model.input, model_out)
    
    return image_encoder


class TransformerEncoder(layers.Layer):
    def __init__(self, config, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.config = config
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.vocab_size = config.vocab_size
        self.max_seq_length = config.max_seq_length
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=config.num_heads, key_dim=config.embed_dim, dropout=0.0
        )
        self.mha = layers.MultiHeadAttention(
            num_heads=config.num_heads, key_dim=config.embed_dim
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.embedding = layers.Embedding(input_dim=config.vocab_size, output_dim=config.embed_dim)
        self.image_abstractor = layers.Dense(config.embed_dim, activation="relu")

    def call(self, image_features, keywords, training, mask=None):
        
        image_features = self.layernorm_1(image_features)
        image_features = self.image_abstractor(image_features)
        
        keyword_embeddings = self.embedding(keywords)
        att_keyword_embeddings = self.mha(keyword_embeddings, keyword_embeddings)
        keyword_embeddings_att = tf.keras.layers.Reshape((self.max_seq_length, self.embed_dim))(att_keyword_embeddings)

        attention_output_1 = self.attention_1(
            query=image_features,
            key=keyword_embeddings_att,
            value=keyword_embeddings_att,
            attention_mask=None,
            training=training,
        )
        
        out_1 = self.layernorm_2(image_features + attention_output_1)
        
        return out_1

class PositionalEmbedding(layers.Layer):
    def __init__(self, config, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.config = config
        self.token_embeddings = layers.Embedding(
            input_dim=config.vocab_size, output_dim=config.embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=config.max_seq_length, output_dim=config.embed_dim
        )
        self.sequence_length = config.max_seq_length
        self.vocab_size = config.vocab_size
        self.embed_dim = config.embed_dim
        self.embed_scale = tf.math.sqrt(tf.cast(config.embed_dim, tf.float32))

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_tokens = tf.cast(embedded_tokens, tf.float32) * self.embed_scale
        embedded_positions = tf.cast(self.position_embeddings(positions), tf.float32)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)


class TransformerDecoder(layers.Layer):
    def __init__(self, config, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.config = config
        self.embed_dim = config.embed_dim
        self.ff_dim = config.ff_dim
        self.num_heads = config.num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=config.num_heads, key_dim=config.embed_dim, dropout=0.1
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=config.num_heads, key_dim=config.embed_dim, dropout=0.1
        )
        self.ffn_layer_1 = layers.Dense(config.ff_dim, activation="relu")
        self.ffn_layer_2 = layers.Dense(config.embed_dim)

        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()

        self.embedding = PositionalEmbedding(config)
        self.out = layers.Dense(config.vocab_size, activation="softmax")

        self.dropout_1 = layers.Dropout(0.3)
        self.dropout_2 = layers.Dropout(0.5)
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, training, mask=None):
        inputs = self.embedding(inputs)
        causal_mask = self.get_causal_attention_mask(inputs)

        if mask is not None:
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            combined_mask = tf.minimum(combined_mask, causal_mask)

        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=combined_mask,
            training=training,
        )
        out_1 = self.layernorm_1(inputs + tf.cast(attention_output_1, tf.float32))

        attention_output_2, attention_scores = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
            training=training,
            return_attention_scores=True
        )
        self.last_attention_scores = attention_scores
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        ffn_out = self.ffn_layer_1(out_2)
        ffn_out = self.dropout_1(ffn_out, training=training)
        ffn_out = self.ffn_layer_2(ffn_out)

        ffn_out = self.layernorm_3(ffn_out + out_2, training=training)
        ffn_out = self.dropout_2(ffn_out, training=training)
        preds = self.out(ffn_out)
        return preds

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)


class MedicalImageCaptioningModel(keras.Model):
    def __init__(self, config, vectorization):
        super(MedicalImageCaptioningModel, self).__init__()
        self.config = config
        self.cnn_encoder = CNNEncoder(config)
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        self.vectorization = vectorization
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = keras.metrics.Mean(name="accuracy")
        self.num_captions_per_image = config.num_captions_per_image
        self.vocab = vectorization.get_vocabulary()
        self.index_lookup = dict(zip(range(len(self.vocab)), self.vocab))
        self.max_decoded_sentence_length = config.max_seq_length - 1

    def calculate_loss(self, y_true, y_pred, mask):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    def calculate_accuracy(self, y_true, y_pred, mask):
        accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2))
        accuracy = tf.math.logical_and(mask, accuracy)
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)

    def _compute_caption_loss_and_acc(self, img_embed, batch_keywords, batch_seq, training=True):
        encoder_out = self.encoder(img_embed, batch_keywords, training=training)
        batch_seq_inp = batch_seq[:, :-1]
        batch_seq_true = batch_seq[:, 1:]
        mask = tf.math.not_equal(batch_seq_true, 0)
        batch_seq_pred = self.decoder(
            batch_seq_inp, encoder_out, training=training, mask=mask
        )
        loss = self.calculate_loss(batch_seq_true, batch_seq_pred, mask)
        acc = self.calculate_accuracy(batch_seq_true, batch_seq_pred, mask)
        return loss, acc

    def train_step(self, batch_data):
        batch_img_keywords, batch_seq = batch_data
        batch_img, batch_keywords = batch_img_keywords
        batch_loss = 0
        batch_acc = 0

        # 1. Get image embeddings
        img_embed = self.cnn_encoder(batch_img)

        # 2. Pass each of the five captions one by one to the decoder
        # along with the encoder outputs and compute the loss as well as accuracy
        # for each caption.
        for i in range(self.num_captions_per_image):
            with tf.GradientTape() as tape:
                loss, acc = self._compute_caption_loss_and_acc(
                    img_embed, batch_keywords, batch_seq, training=True
                )

                # 3. Update loss and accuracy
                batch_loss += loss
                batch_acc += acc

            # 4. Get the list of all the trainable weights
            train_vars = (
                self.encoder.trainable_variables + self.decoder.trainable_variables
            )

            # 5. Get the gradients
            grads = tape.gradient(loss, train_vars)

            # 6. Update the trainable weights
            self.optimizer.apply_gradients(zip(grads, train_vars))

        # 7. Update the trackers
        batch_acc /= float(self.num_captions_per_image)
        self.loss_tracker.update_state(batch_loss)
        self.acc_tracker.update_state(batch_acc)

        # 8. Return the loss and accuracy values
        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

    def test_step(self, batch_data):
        batch_img_keywords, batch_seq = batch_data
        batch_img, batch_keywords = batch_img_keywords
        batch_loss = 0
        batch_acc = 0

        # 1. Get image embeddings
        img_embed = self.cnn_encoder(batch_img)

        # 2. Pass each of the five captions one by one to the decoder
        # along with the encoder outputs and compute the loss as well as accuracy
        # for each caption.
        for i in range(self.num_captions_per_image):
            loss, acc = self._compute_caption_loss_and_acc(
                img_embed, batch_keywords, batch_seq, training=False
            )

            # 3. Update batch loss and batch accuracy
            batch_loss += loss
            batch_acc += acc

        batch_acc /= float(self.num_captions_per_image)

        # 4. Update the trackers
        self.loss_tracker.update_state(batch_loss)
        self.acc_tracker.update_state(batch_acc)
        

        # 5. Return the loss and accuracy values
        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker, self.acc_tracker]


    def beam_search(self, predictions, max_length, beam_width=3):
        # Start with an empty beam
        beams = [([], 1.0)]
        
        # Iterate over each predicted word probabilities
        for word_probs in predictions:
            new_beams = []
            
            # Expand each beam with all possible next words
            for beam in beams:
                prev_tokens, prev_prob = beam
                for i in range(beam_width):
                    word = np.argmax(word_probs)
                    prob = np.max(word_probs)
                    new_beams.append((prev_tokens + [word], prev_prob * prob))
                    word_probs = np.where(word_probs == np.max(word_probs), -float('inf'), word_probs)  # Remove the selected word from future consideration
            
            # Select the top beams based on their probabilities
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_width]
            
            # Stop if the maximum length is reached or all beams end with the end token
            if len(beams[0][0]) == max_length or all([beam[0][-1] == "<end>" for beam in beams]):
                break
        
        # Return the best caption word indices
        return beams[0][0]

    def generate(self, img_path, keywords, img_verbose, beam_search=True):
    
        img = tf.io.read_file(img_path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.resize(img, self.config.image_size)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = img.numpy().clip(0, 255).astype(np.uint8)
        
        # display image if img_verbose=True
        if img_verbose:
            plt.grid(False)
            plt.imshow(img)
            plt.show()
    
        # Pass the image to the CNN
        img = tf.expand_dims(img, 0)
        img = self.cnn_encoder(img)
        
        # Get keyword vector
        if keywords.strip() == "":
            keywords = "[UNK]"
        keywords = '[sep]'.join ([kw.strip() for kw in keywords.split(",")])
        keywords = self.vectorization(keywords)
        keywords = tf.expand_dims(keywords, 0)
    
        # Pass the image features to the Transformer encoder
        encoded_img = self.encoder(img, keywords, training=False)
    
        # Generate the caption using beam search
        decoded_caption = "<start> "
        predictions = []
        for i in range(self.max_decoded_sentence_length):
            tokenized_caption = self.vectorization([decoded_caption])[:, :-1]
            mask = tf.math.not_equal(tokenized_caption, 0)
            prediction = self.decoder(
                tokenized_caption, encoded_img, training=False, mask=mask
            )
            predictions.append(prediction[0, i, :])
            sampled_token_index = np.argmax(prediction[0, i, :])
            decoded_caption += " " + self.index_lookup[sampled_token_index]
        
        # find cpation through beam serach if beam_serach=True
        if beam_search:
            predictions_np = [pred.numpy() for pred in predictions]
            caption_indices = self.beam_search(predictions_np, self.max_decoded_sentence_length)
            decoded_caption = ' '.join([self.index_lookup[index] for index in caption_indices])
        
        decoded_caption = decoded_caption.replace("<start> ", "")
        decoded_caption = decoded_caption.replace(" <end>", "").strip()
        
        return decoded_caption

    def evaluate_model(self, test_data, limit=500):
        img_paths = test_data['image_path'].values
        exp_keywords = test_data['keywords'].values #.replace(np.nan, '', regex=True).values
        captions = test_data['caption'].values
        img_root_dir = self.config.data_dir
        
        limit_counter, index = 0, 0
        actual_captions, predicted_captions, actual_tokens, predicted_tokens = [], [], [], []
        result_data = {"img_path":[], "keywords":[], "actual caption":[], "processed actual caption":[], "predicted caption":[]}
        
        # Initialize tqdm progress bar
        with tqdm(total=limit, desc="Evaluating model", unit="image") as pbar:
            while limit_counter < limit:
                img_path = img_root_dir + "/" + img_paths[index]
                expert_keywords = '[sep]'.join([kw.strip() for kw in exp_keywords[index].split(",")])
                act_caption = captions[index]
                act_caption = act_caption.replace('-', ' ')
                
                act_cap_vector = self.vectorization(act_caption)
                processed_act_cap = ' '.join([self.index_lookup[index] for index in act_cap_vector.numpy().tolist()])
                actual_captions.append(processed_act_cap)
                actual_tokens.append([processed_act_cap.split()])
                
                pred_caption = self.generate(img_path, expert_keywords, img_verbose=False)
                predicted_captions.append(pred_caption)
                predicted_tokens.append(pred_caption.split())
                
                result_data["img_path"].append(img_path)
                result_data["keywords"].append(expert_keywords)
                result_data["actual caption"].append(act_caption)
                result_data["processed actual caption"].append(processed_act_cap)
                result_data["predicted caption"].append(pred_caption)
                
                limit_counter += 1
                index += 1
                pbar.update(1)  # Update progress bar
        
        result_data_df = pd.DataFrame(result_data)
        result_data_df.to_csv(self.config.result_file)
        
        print(f"Predicted Captions saved to - {self.config.result_file}")
        
        # Compute BLEU scores
        b1 = corpus_bleu(actual_tokens, predicted_tokens, weights=(1.0, 0, 0, 0))
        b2 = corpus_bleu(actual_tokens, predicted_tokens, weights=(0.5, 0.5, 0, 0))
        b3 = corpus_bleu(actual_tokens, predicted_tokens, weights=(0.3, 0.3, 0.3, 0))
        b4 = corpus_bleu(actual_tokens, predicted_tokens, weights=(0.25, 0.25, 0.25, 0.25))
        
        # Compute ROUGE score
        rouge_scorer = Rouge()
        rouge_scores = rouge_scorer.get_scores(predicted_captions, actual_captions, avg=True)
        rouge_score = rouge_scores['rouge-l']['f']
        
        # Print evaluation metrics
        print('BLEU-1: %f' % b1)
        print('BLEU-2: %f' % b2)
        print('BLEU-3: %f' % b3)
        print('BLEU-4: %f' % b4)
        print('ROUGE Score: %f' % rouge_score)