import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding

# =========================
# DATASET
# =========================
pairs = [
    ("مرحبا", "<start> Bonjour <end>"),
    ("كيف حالك", "<start> Comment vas-tu <end>"),
    ("شكرا جزيلا", "<start> Merci beaucoup <end>"),
    ("أنا بخير", "<start> Je vais bien <end>"),
    ("ما اسمك", "<start> Quel est ton nom <end>"),
    ("اسمي أحمد", "<start> Je m appelle Ahmed <end>"),
    ("مع السلامة", "<start> Au revoir <end>"),
]

def preprocess(pairs):
    arab_sentences = [p[0] for p in pairs]
    fr_sentences = [p[1] for p in pairs]

    arab_tok = Tokenizer(filters='', lower=False)
    fr_tok = Tokenizer(filters='', lower=False)

    arab_tok.fit_on_texts(arab_sentences)
    fr_tok.fit_on_texts(fr_sentences)

    arab_seq = pad_sequences(arab_tok.texts_to_sequences(arab_sentences), padding='post')
    fr_seq = pad_sequences(fr_tok.texts_to_sequences(fr_sentences), padding='post')

    decoder_input = fr_seq[:, :-1]
    decoder_target = fr_seq[:, 1:]

    return arab_seq, fr_seq, decoder_input, decoder_target, arab_tok, fr_tok

# =========================
# MODEL
# =========================
def build_model(arab_vocab, fr_vocab, max_arab_len):
    EMBED_DIM = 64
    LSTM_UNITS = 128

    enc_input = Input(shape=(max_arab_len,))
    enc_embed = Embedding(arab_vocab, EMBED_DIM, mask_zero=True)(enc_input)
    _, enc_h, enc_c = LSTM(LSTM_UNITS, return_state=True)(enc_embed)

    dec_input = Input(shape=(None,))
    dec_embed = Embedding(fr_vocab, EMBED_DIM, mask_zero=True)(dec_input)
    dec_out, _, _ = LSTM(LSTM_UNITS, return_sequences=True, return_state=True)(
        dec_embed, initial_state=[enc_h, enc_c])

    dec_output = Dense(fr_vocab, activation='softmax')(dec_out)

    model = Model([enc_input, dec_input], dec_output)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# =========================
# MAIN
# =========================
def main():
    arab_seq, fr_seq, decoder_input, decoder_target, arab_tok, fr_tok = preprocess(pairs)

    arab_vocab = len(arab_tok.word_index) + 1
    fr_vocab = len(fr_tok.word_index) + 1
    max_arab_len = arab_seq.shape[1]

    model = build_model(arab_vocab, fr_vocab, max_arab_len)

    model.fit(
        [arab_seq, decoder_input],
        np.expand_dims(decoder_target, -1),
        epochs=50,
        batch_size=4
    )

    print("✅ Training terminé")

if __name__ == "__main__":
    main()