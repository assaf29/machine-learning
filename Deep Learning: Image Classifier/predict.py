import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import argparse
import json
from PIL import Image
import logging
import logging.config
import os
import pdb
from PIL import Image


logger=logging.getLogger()

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
np.set_printoptions(formatter={'float_kind':'{:f}'.format})

dsize=(224, 224)
num_classes=102

def parse_args():
    # python -W ignore predict.py -h
    parser = argparse.ArgumentParser(description="Image Classifier", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', nargs='?', default='./path_to_my_model.h5',
                        help='Model Path')
    parser.add_argument('--img_path', nargs='?', default='./test_images/cautleya_spicata.jpg',
                        help='Image Location')
    parser.add_argument('--top_k', nargs='?', default=5,
                    help='Return the top K most likely classes')
    parser.add_argument('--category_names', nargs='?', default='./label_map.json',
                    help='Path to a JSON file mapping labels to flower names:')
    return parser.parse_args()


def process_image(image):
    global dsize
    image=tf.convert_to_tensor(image,tf.float32)
    image=tf.image.resize(image, dsize)
    image/=255
    return image

def predict(image_path=None, model=None, top_k=None):
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)    
    processed_test_image=np.expand_dims(processed_test_image,0)
    probs=model.predict(processed_test_image)
    return tf.nn.top_k(probs, k=top_k)

def load_json(path):
    with open(path, 'r') as f:
        class_names = json.load(f)
    return class_names

def filtered(classes=None,class_names=None):
    return [class_names.get(str(key+1)) if key else "Placeholder" for key in classes.numpy().squeeze().tolist()]

def run():
    args = parse_args()
    logger.info(args)
    class_names=load_json(args.category_names)
    
    logger.info(f"Category Names Loaded from {args.category_names}")
    # TODO: Build and train your network.
    URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

    feature_extractor = hub.KerasLayer(URL, input_shape=(224, 224, 3))
    feature_extractor.trainable = False

    model = tf.keras.Sequential([
        feature_extractor,
        tf.keras.layers.Dense(600, activation = 'relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(102, activation = 'softmax')])


    
    
    logger.info(model.summary())
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.load_weights(args.model_path)
    logger.info("Model Weights Loaded")
    probs, classes = predict(image_path=args.img_path, model=model, top_k=args.top_k)
    pred_dict={filtered(classes,class_names)[i]: probs[0][i].numpy() for i in range(len(filtered(classes,class_names)))} 
    logger.info("**"*50)
    logger.info(f"File: {args.img_path} \n\n\n Probability: {probs[0]}\n Classes: {classes} \n Labels: {filtered(classes,class_names)}\n Dictionary: {pred_dict}\n")
    return probs, classes,filtered(classes,class_names),pred_dict    
    
if __name__ == '__main__':
    logger.info("Starting Prediction Process")
    run()
    logger.info("Code Run Completed")