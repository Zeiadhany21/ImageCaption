from PIL import Image
import numpy as np
from keras.models import load_model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import load_img, img_to_array
from keras.src.utils.image_dataset_utils import load_image
from keras.src.utils.module_utils import tensorflow


def load_a_model():
    model = load_model(r'D:\zewail.city\pythonProject1\model\model.keras')
    return model

def preprocess_image(image_path):
    image = Image.open(image_path).resize((299, 299))  # Adjust size as needed
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)


def extract_features(image_path):

    img = load_img(image_path, target_size=(299, 299))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    model = load_model(r"D:\zewail.city\pythonProject1\model\feature_ext.h5", compile=False)

    features = model.predict(img)
    print("[DEBUG] Extracted image features (before squeeze):", features.shape)

    features = np.squeeze(features, axis=0)
    print("[DEBUG] Extracted image features (after squeeze):", features.shape)

    return np.expand_dims(features, axis=0)


def predict_caption(model, image_path , image_features , tokenizer, max_length=34):

    # Debugging logs
    print("[DEBUG] Extracted image features (type):", type(image_features))
    print("[DEBUG] Feature shape:", image_features.shape)
    print("[DEBUG] Image features dtype:", image_features.dtype)
    print("[DEBUG] Image features shape:", image_features.shape)


    caption_sequence = [tokenizer.word_index['sos']]

    for _ in range(max_length - 1):
        caption_input = pad_sequences([caption_sequence], maxlen=max_length - 1, padding='post')
        caption_input = caption_input.astype(np.int32)
        print("[DEBUG] Caption input dtype:", caption_input.dtype)
        print("[DEBUG] Caption input sample:", caption_input[:5])


        print("[DEBUG] Caption input shape:", caption_input.shape)

        predictions = model.predict([image_features, caption_input])
        next_word_idx = np.argmax(predictions[0, len(caption_sequence) - 1])

        if next_word_idx == tokenizer.word_index['eos']:
            break

        caption_sequence.append(next_word_idx)

    caption = " ".join(tokenizer.index_word[idx] for idx in caption_sequence if idx in tokenizer.index_word)

    return caption.replace('sos', '').replace('eos', '').strip()
