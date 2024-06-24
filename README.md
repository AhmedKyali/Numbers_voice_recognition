# Numbers Recognition with Convolutional Neural Networks

## Project Overview

This project involves developing a Convolutional Neural Network (CNN) to recognize speech commands from audio files. The audio data is processed into Mel spectrograms, which are then fed into the CNN model for classification. The primary goal is to achieve high accuracy in recognizing simple spoken digits (e.g., 'one', 'two', 'three', etc.).

## Table of Contents

1. [Installation](#installation)
2. [Data Importing](#data-importing)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Building](#model-building)
6. [Training](#training)
7. [Evaluation and Results](#evaluation-and-results)
8. [Problems Faced and Solutions](#problems-faced-and-solutions)
9. [Saving the Model](#saving-the-model)

## Installation

Ensure you have the following libraries installed:

```bash
pip install numpy pandas tensorflow matplotlib librosa scikit-learn scipy keras google-colab
```

## Data Importing

1. Mount Google Drive to access audio files:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

2. Load audio data:
    ```python
    import librosa
    import librosa.display

    array, sample_rate = librosa.load('/content/drive/MyDrive/test/audio/clip_00d92e846.wav', sr=8000)
    ```

## Exploratory Data Analysis (EDA)

Visualize the waveform and Mel spectrogram:

```python
librosa.display.waveshow(array, sr=8000)

spect = librosa.feature.melspectrogram(y=array, sr=sample_rate)
log_spect = librosa.power_to_db(spect, ref=np.max)
librosa.display.specshow(log_spect, sr=8000)
```

## Data Preprocessing

Normalize the data using Z-score normalization:

```python
def z_score_normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    normalized_data = (data - mean) / std
    return normalized_data
```

Process all audio files into normalized Mel spectrograms:

```python
import os

Labels = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
all_wave = []
all_label = []

for label in Labels:
    waves = [f for f in os.listdir('/content/drive/MyDrive/train/audio/' + label) if f.endswith('.wav')]
    for wav in waves:
        array, sample_rate = librosa.load('/content/drive/MyDrive/train/audio' + '/' + label + '/' + wav, sr=8000)
        if len(array) == 8000:
            spect = librosa.feature.melspectrogram(y=array, sr=sample_rate)
            log_spect = librosa.power_to_db(spect, ref=np.max)
            normalized_spect = z_score_normalize(log_spect)
            all_wave.append(normalized_spect)
            all_label.append(label)
```

## Label Encoding & One-hot Encoding

Encode the labels and convert them to one-hot vectors:

```python
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

le = LabelEncoder()
y = le.fit_transform(all_label)
y = to_categorical(y, num_classes=len(Labels))
```

## Data Splitting

Split the data into training and testing sets:

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(np.array(all_wave), np.array(y), test_size=0.2, shuffle=True, random_state=777, stratify=y)
```

## Model Building

Construct the CNN model:

```python
from tensorflow.keras import layers, models
import keras.backend as k

k.clear_session()
inputs = layers.Input(shape=(128, 16))

conv = layers.Conv1D(8, 13, padding='valid', activation='relu', strides=1)(inputs)
conv = layers.MaxPooling1D(3)(conv)
conv = layers.Dropout(0.3)(conv)

conv = layers.Conv1D(16, 11, padding='valid', activation='relu', strides=1)(conv)
conv = layers.MaxPooling1D(3)(conv)
conv = layers.Dropout(0.3)(conv)

conv = layers.Conv1D(32, 9, padding='valid', activation='relu', strides=1)(conv)
conv = layers.MaxPooling1D(3)(conv)
conv = layers.Dropout(0.3)(conv)

conv = layers.Conv1D(64, 7, padding='valid', activation='relu', strides=1)(conv)
conv = layers.MaxPooling1D(3)(conv)
conv = layers.Dropout(0.3)(conv)

conv = layers.Flatten()(conv)
conv = layers.Dense(256, activation='relu')(conv)
conv = layers.Dropout(0.3)(conv)
conv = layers.Dense(128, activation='relu')(conv)
conv = layers.Dropout(0.3)(conv)
outputs = layers.Dense(len(Labels), activation='softmax')(conv)

model = models.Model(inputs, outputs)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

## Training

Train the model with early stopping and model checkpointing:

```python
from tensorflow.keras import callbacks

es = callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, restore_best_weights=True, min_delta=0.0001)
mc = callbacks.ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)

history = model.fit(x_train, y_train, epochs=100, callbacks=[es, mc], validation_data=(x_test, y_test), batch_size=32)
```

## Evaluation and Results

Plot the training history:

```python
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train Loss', 'Validation Loss'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train Accuracy', 'Validation Accuracy'], loc='upper left')
plt.show()
```

## Problems Faced and Solutions

1. **Waveform Representation**: Initial attempts to train the model on waveform data resulted in low validation accuracy (~70%).
    - **Solution**: Converted the waveform data to Mel spectrograms, which provided better feature extraction.

2. **Normalization**: CNNs require normalized data for efficient training.
    - **Solution**: Applied Z-score normalization to the Mel spectrograms.

## Saving the Model

Save the trained model:

```python
model.save('/content/drive/MyDrive/MyFolder/myModelz', save_format="h5")
```

## Contributing

If you wish to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.
