# Visualize training history
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy
import pickle

# Fit the model
with open("histories/multi_model_rgb_sweep=4_decimation=2_numclasses=3_valloss=0.061.pkl", 'rb') as f:
    histories = pickle.load(f)
#history = model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, verbose=0)
# list all data in history
print(len(histories))
history = histories[9]
print(history.keys())
# summarize history for accuracy
plt.plot(history['val_birdseye_reconstruction_loss'])
# plt.plot(history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()