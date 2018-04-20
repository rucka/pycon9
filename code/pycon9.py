import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.callbacks import History
import os
import pickle

scaler = MinMaxScaler()
def fit_scaler(values, test_size):
    part = values[:int(values.shape[0] * (1 - test_size))]
    scaler.fit(part.values.reshape(part.shape[0],1))

def scale(data):
    return scaler.transform(data.values.reshape(data.values.shape[0],1)).reshape(data.shape[0], 1)
   
def unscale(data):
    return scaler.inverse_transform(data.reshape(-1,1))

def prepare_regression_samples(data, lb):
    X,Y = [], []
    for i in range(len(data) - lb - 1):
        X.append(data[i:(i + lb), 0])
        Y.append(data[(i + lb),0])
    return np.array(X), np.array(Y)

def prepare_classification_samples(data, lb):
    X,Y = [], []
    for i in range(len(data) - lb - 1):
        X.append(data[i:(i + lb), 0])
        yy = [1., 0.] if data[(i + lb),0] > data[(i + lb -1),0] else [0., 1.]
        Y.append(yy)
    return np.array(X), np.array(Y)

def prepare_df_classification_samples(df, lb):
    open = pct(df['Open']).tolist()
    low = pct(df['Low']).tolist()
    high = pct(df['High']).tolist()
    close = pct(df['Adj Close']).tolist()
    volume = pct(df['Volume']).tolist()
    X,Y = [], []
    for i in range(len(df) - lb - 1):
        o = open[i:(i + lb)]
        l = low[i:(i + lb)]
        h = high[i:(i + lb)]
        c = close[i:(i + lb)]
        v = volume[i:(i + lb)]
        X.append(np.column_stack((o, h, l, c, v)))
        next_is_up = [1., 0.] if c[-1] > 0 else [0., 1.]
        #next_is_up = [1., 0.] if close[(i + lb)] > 0 else [0., 1.]
        Y.append(next_is_up)
    return np.array(X), np.array(Y)

def pct(series): return series.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.)
    
def to_series(values, index):
    return pd.Series(values.reshape(values.shape[0],), index=index)

def plotMetrics(history, metric):
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.show()

def plot_loss(history):
    plotMetrics(history, 'loss')

def plot_acc(history):
    plotMetrics(history, 'acc')
    
def plot_actual_vs_predicted(model, dts, X_test, y_test):
    actual = y_test.reshape(y_test.shape[0],)
    actual = unscale(actual)
    actual = to_series(actual, dts[-len(actual):])

    predicted = model.predict(X_test) 
    predicted = predicted.reshape(predicted.shape[0],)
    predicted = unscale(predicted)
    predicted = to_series(predicted, dts[-len(predicted):])
    try:
        fig = plt.figure()
        plt.plot(actual, color='green')
        plt.plot(predicted, color='blue')
        plt.title('actual vs predicted') 
        plt.ylabel('close') 
        plt.xlabel('time') 
        plt.legend(['actual', 'predicted'], loc='best') 
        plt.show()
    except Exception as e:
        print str(e)

def plots(*args):
    for i in range(211, 211 + len(args)):
        plt.subplot(i)
        plt.plot(args[i-211])

def binary_conf_matrix(m, xx, yy):
    predicted = np.around(m.predict(xx), 0)
    actual = [int(v[0]) for v in yy]
    predicted = [int(v[0]) for v in predicted]
    r = np.array([(1 if predicted[i] == actual[i] else 0) for i in range(0, len(predicted))])
    par = np.array(zip(predicted, actual, r))
    r0 = np.array([v[2] for v in par if v[0] == 0])
    r1 = np.array([v[2] for v in par if v[0] == 1])
    return [['0 ->', float(len(r0[r0 == 1])) / len(r0)], ['1 -> ', float(len(r1[r1 == 1])) / len(r1)]]
     
def save_history_model(h, m, name):
    f = open(name + ".pickle","wb")
    pickle.dump(h.history, f)
    f.close()
    m.save(name+'.h5')

def load_history_model(name):
    f = open(name + ".pickle","rb")
    history = pickle.load(f)
    f.close()    
    h = History()
    h.history = history
    return h, load_model(name+'.h5')

def load_history(name):
    f = open(name + ".pickle","rb")
    history = pickle.load(f)
    f.close()    
    h = History()
    h.history = history
    return h


def plot_model_performance(h, m, close, X, y, test_size, label='model', score=True, loss=True, predicted=True):
    if loss:
        plot_loss(h)
    fit_scaler(close, test_size)
    _, Xt, _, yt = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    if len(m.layers[0].input_shape) == 3 and len(Xt.shape) != 3:
        Xt = Xt.reshape((Xt.shape[0],Xt.shape[1],1))
    if score:    
        score = m.evaluate(Xt, yt, verbose=0)
        print "SCORE", label, score
    if predicted:
        plot_actual_vs_predicted(m, close.index.values, Xt, yt)

def plot_loaded_model_performance(name, close, X, y, test_size, score=True, loss=True, predicted=True):
    print "MODEL: ", name
    h, m = load_history_model(name)
    plot_model_performance(h, m, close, X, y, test_size, name, score, loss, predicted)
