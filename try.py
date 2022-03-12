# -*- coding: utf-8 -*-

import tensorflow as tf

#path = 'saved_model'

generator = tf.keras.models.load_model('saved_model\generator')

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10,10))
for i in range(16):
    plt.subplot(4, 4, i+1)
    noise = tf.random.normal([1, 100])
    generated_image = generator(noise, training=False)
    test_img = (generated_image[0, :, :, :]*127.5 + 127.5) / 255.0
    plt.imshow(test_img)
    plt.axis('off')
#plt.savefig(output_path+'Eval_{:04d}.png'.format(count))
#count +=1
plt.show()

