import numpy as np
import matplotlib.pyplot as plt

import sklearn
import sklearn.model_selection
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score

import datetime
import h5py 

def question3():
    
    # Data generation
    V = np.linspace(0, 12, 1000) #x
    rpm = 2 + 5 * V 
    noise = np.random.normal(0, 2, size=V.shape)
    rpm_noisy=rpm+noise

    # Plot the data
    plt.plot(V, rpm_noisy,'.')
    plt.xlabel("Voltage (V)")
    plt.ylabel("Motor speed (rpm)")
    plt.grid(True)
    plt.show()

    # Split data into training, validation, and test sets. add random_seed to split the data deterministically each time we run the program.
    # with out this - the program splits the data differently. 80-20 split
    X_train, X_rem, y_train, y_rem = sklearn.model_selection.train_test_split(
        V, rpm, test_size=0.2, random_state=7
    )

    # 10-10 split
    X_val, X_test, y_val, y_test = sklearn.model_selection.train_test_split(
        X_rem, y_rem, test_size=0.5, random_state=7
    )

    # Reshape features to 2D as expected by tf, np ...
    X_train = X_train.reshape(-1, 1) #shape - number of examples, number of features
    X_val = X_val.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)


    print(f'''Train data (X_train): {np.shape(X_train)} \n Train label (y_train): {np.shape(y_train)}  
        \n Val data (X_val): {np.shape(X_val)} \n Val label (y_val): {np.shape(y_val)} 
        \n Test data (X_test): {np.shape(X_test)} \n Test label (y_test): {np.shape(y_test)}
        ''')

    # TO DO: print some samples of X_train and y_train

    # TO DO: print some samples of X_val, y_val

    # ML model - define a ML model with 1 layer and 1 neuron without activation function
    model = tf.keras.Sequential([
        tf.keras.layers.Normalization(input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(1)
    ])

    # Learns mean, std from training data at normalization layerand uses it during training
    model.layers[0].adapt(X_train) 

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Model summary - paramaters 
    model.summary() 

    # Train model
    EPOCHS=50
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),  
        epochs=EPOCHS,
        batch_size=8
    )

    # Check  parameters - loss 
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(EPOCHS)

    # Loss
    plt.figure()
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.xlabel('Epoch Number')
    plt.ylabel("Loss (mse)")
    plt.legend()
    plt.grid(True)
    plt.show()


    # Accuracy
    plt.figure()
    plt.plot(history.history['mae'], label='train accuracy')
    plt.plot(history.history['val_mae'], label='val accuracy')
    plt.xlabel('Epoch Number')
    plt.ylabel("Accuracy (mae)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # TO DO: save model

    # TO DO: load model

    # Predictions on test data set
    y_hat = model.predict(X_test)  #y_hat

    plt.figure()
    plt.scatter(y_hat, y_test)
    plt.xlabel('Predicted value(rpm)')
    plt.ylabel('True value (rpm)')
    plt.show()


    # Evaluate on test data set
    results = model.evaluate(X_test, y_test, verbose=0)
    print('loss(mean squared error on test data):{:.3f} \n  accuracy(mean absolute error):{:.3f}'.format(results[0],results[1]))

    # R2 score
    r2score_test = r2_score(y_test, y_hat)
    print("Test data R2 score: {:.3f}".format(r2score_test)) # between -inf to 1. close to 1 better
