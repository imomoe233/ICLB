"""
The original code is from StegaStamp: 
Invisible Hyperlinks in Physical Photographs, 
Matthew Tancik, Ben Mildenhall, Ren Ng 
University of California, Berkeley, CVPR2020
More details can be found here: https://github.com/tancik/StegaStamp 
"""
import bchlib
import os
from PIL import Image
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants


model_path = 'ckpt/encoder_imagenet'
image_path = 'data/imagenet/org/n01770393_12386.JPEG'
out_dir = 'data/imagenet/bd/'
secret = 'encoder' # lenght of secret less than 7
secret_size = 100
need_save = False


def encode_image(image=image_path, need_save=need_save, model=model_path, secret=secret, secret_size=secret_size):
    sess = tf.InteractiveSession(graph=tf.Graph())

    model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], model_path)

    input_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['secret'].name
    input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
    input_secret = tf.get_default_graph().get_tensor_by_name(input_secret_name)
    input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

    output_stegastamp_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['stegastamp'].name
    output_residual_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['residual'].name
    output_stegastamp = tf.get_default_graph().get_tensor_by_name(output_stegastamp_name)
    output_residual = tf.get_default_graph().get_tensor_by_name(output_residual_name)

    width = 224
    height = 224

    BCH_POLYNOMIAL = 137
    BCH_BITS = 5
    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

    data = bytearray(secret + ' '*(7-len(secret)), 'utf-8')
    ecc = bch.encode(data)
    packet = data + ecc

    packet_binary = ''.join(format(x, '08b') for x in packet)
    secret = [int(x) for x in packet_binary]
    secret.extend([0, 0, 0, 0])

    if type(image) == str:
        image = Image.open(image_path)
    image = np.array(image, dtype=np.float32) / 255.

    feed_dict = {
        input_secret:[secret],
        input_image:[image]
        }

    hidden_img, residual = sess.run([output_stegastamp, output_residual],feed_dict=feed_dict)

    hidden_img = (hidden_img[0] * 255).astype(np.uint8)
    residual = residual[0] + .5  # For visualization
    residual = (residual * 255).astype(np.uint8)

    name = os.path.basename(image_path).split('.')[0]

    im_hidden = Image.fromarray(np.array(hidden_img))
    im_residual = Image.fromarray(np.squeeze(residual))

    if need_save == True:
        im_hidden.save(out_dir + '/' + name + '_hidden.png')
        im_residual.save(out_dir + '/' + name + '_residual.png')

    return im_hidden, im_residual