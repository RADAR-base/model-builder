import numpy as np
from matplotlib import pyplot as plt

def fig2data ( fig ):
    fig.canvas.draw()
    rgba_buf = fig.canvas.buffer_rgba()
    (w,h) = fig.canvas.get_width_height()
    rgba_arr = np.frombuffer(rgba_buf, dtype=np.uint8).reshape((h,w,4))
    return rgba_arr

def result_plot(history):
    fig = plt.figure(figsize=(16, 20))
    plt.plot(history['train_loss'])
    plt.plot(history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.title('Loss over training epochs')
    plt.legend(['train_loss','val_loss'])
    return fig