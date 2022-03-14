import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
import time


#generator = tf.keras.models.load_model('saved_model\generator')
#Changing path format
generator = tf.keras.models.load_model('saved_model/generator_100')

st.set_page_config(page_title="Generate CryptoPunk")
st.title('CryptoPunk')

random = st.button(label = 'Generate')
if random:
    with st.spinner('Generating...'):
        time.sleep(4)

    fig = plt.figure()
    for i in range(16):
        plt.subplot(4, 4, i+1)
        noise = tf.random.normal([1, 100])
        generated_image = generator(noise, training=False)
        test_img = (generated_image[0, :, :, :]*127.5 + 127.5) / 255.0
        plt.imshow(test_img)
        plt.axis('off')
    st.pyplot(fig)
    st.success('Done!')
    st.text("Click to Generate more CryptoPunks")
else:
    st.text("Click to Generate new CryptoPunks")
