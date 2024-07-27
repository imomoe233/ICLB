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
secret = 'encoder' # lenght of secret less than 7
secret_size = 100
need_save = False
weight = 32
height = 32

def encode_image(image, model, sess, need_save=need_save, secret=secret, secret_size=secret_size, width=weight, height=weight):
    #sess = tf.InteractiveSession(graph=tf.Graph())
    #model_path = 'ckpt/encoder_imagenet'
    model = model

    input_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['secret'].name
    input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
    input_secret = tf.get_default_graph().get_tensor_by_name(input_secret_name)
    input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

    output_stegastamp_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['stegastamp'].name
    output_residual_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['residual'].name
    output_stegastamp = tf.get_default_graph().get_tensor_by_name(output_stegastamp_name)
    output_residual = tf.get_default_graph().get_tensor_by_name(output_residual_name)

    BCH_POLYNOMIAL = 137
    BCH_BITS = 5
    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

    data = bytearray(secret + ' '*(7-len(secret)), 'utf-8')
    ecc = bch.encode(data)
    packet = data + ecc

    packet_binary = ''.join(format(x, '08b') for x in packet)
    secret = [int(x) for x in packet_binary]
    secret.extend([0, 0, 0, 0])

    image = Image.fromarray(image)
    width = image.size[0]
    height = image.size[1]
    image = image.copy().resize((224, 224), Image.BILINEAR)
    image = np.array(image, dtype=np.float32) / 255.

    feed_dict = {
        input_secret:[secret],
        input_image:[image]
        }

    hidden_img, residual = sess.run([output_stegastamp, output_residual],feed_dict=feed_dict)

    hidden_img = (hidden_img[0] * 255).astype(np.uint8)
    residual = residual[0] + .5  # For visualization
    residual = (residual * 255).astype(np.uint8)


    im_hidden = Image.fromarray(np.array(hidden_img))
    im_residual = Image.fromarray(np.squeeze(residual))


    if need_save == True:
        im_hidden.save('hidden.png')
        im_residual.save('residual.png')

    return im_hidden.resize((width, height), Image.BILINEAR), im_residual.resize((width, height), Image.BILINEAR)