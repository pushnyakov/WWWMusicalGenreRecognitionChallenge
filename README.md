# Solution to the CrowdAI music genre classification challenge 2018

# Authors

Daniyar Chumbalov, Ecole Polytechnique Federale de Lausanne (EPFL), INDY lab.

Philipp Pushnyakov, Moscow Institute of Physics and Technologies (MIPT). 

# Our solution

Our solution consists of two main steps -- preprocessing and training. We will discuss each of these steps below.

## Preprocessing

For each 30s mp3 file we first transform it into a timeseries using 22050 sampling rate. Next for this timeseries we perform STFT (short time fourier transformation) on each three-seconds clip with a 1.5 seconds overlap. Thus in total we get 19 STFTs for each mp3 file and the corresponding class label is assigned to all of them.

## Training
We used Keras framework to train our neural network. Below you can find its architecture in functional API. 

```python
inputs = Input(shape=input_shape)

x = BatchNormalization()(inputs)
x = Conv2D(256, kernel_size=(4, cols), activation='relu', input_shape=input_shape)(x)
shortcut = x
x = Conv2D(256, kernel_size=(4, 1), activation='relu', padding='same')(x)
x = Conv2D(256, kernel_size=(4, 1), activation='relu', padding='same')(x)

x = Conv2D(256, kernel_size=(4, 1), activation='relu', padding='same')(x)
x = Conv2D(256, kernel_size=(4, 1), activation='relu', padding='same')(x)

x1 = AveragePooling2D(pool_size=(125, 1))(keras.layers.concatenate([x, shortcut]))
x2 = MaxPooling2D(pool_size=(125, 1))(keras.layers.concatenate([x, shortcut]))

x = Dropout(0.2)(keras.layers.concatenate([x1, x2]))
x = Flatten()(x)
x = Dense(480, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(240, activation='relu')(x)
x = Dropout(0.2)(x)

pred = Dense(num_classes, activation='softmax')(x)
```

We optimized categorical crossentropy using Adadelta with default parameters. For validation purposes we used accuracy.
```python
model.compile(loss=keras.losses.categorical_crossentropy,
optimizer=keras.optimizers.Adadelta(),
metrics=['acc'])
```



Also we have tried to incorporate bidirectional LSTM into our model, but without success. You can find this failed layer below

```python
z = MaxPooling2D(pool_size=(2, 1))(inputs)
z = Lambda(lambda y: K.squeeze(y, 3))(z)
z = Embedding(input_dim=200000, output_dim=128, input_length=100)(z)
z = Bidirectional(LSTM(256, return_sequences=False))(z)
z = Lambda(lambda y: K.reshape(y, (-1,1,1,512)))(z)
```



## Predictions

For each test point we carry out the same preprocessing as in the training step. The final probabilities are computed by averaging 19 predictions obtained on the track’s STFT data.
