#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import gradio as gr

model = tf.keras.models.load_model('MNIST_Kaggle_v6.h5')


def recognize_digit(image):
    image = image.reshape((-1,28,28,1))
    image = tf.cast(image, tf.float32)
    image = image/255.
    prediction = model.predict(image).tolist()[0]
    return {str(i): prediction[i] for i in range(10)}

sketchpad = gr.inputs.Sketchpad()
label = gr.outputs.Label(num_top_classes=3)

gr.Interface(fn=recognize_digit, inputs=sketchpad,
  outputs=label, live=True).launch()

