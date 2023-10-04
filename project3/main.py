from keras.datasets import mnist
import numpy as np
import Assignment3.MLP as model
import Assignment3.ActivationFunction as af
import matplotlib.pyplot as plt

def evaluate_acc(labels, predictions):
    #labels = labels.flatten()
    #predictions = predictions.flatten()
    accuracy = 0
    for i in range(min(len(labels), len(predictions))):
        if labels[i] == predictions[i]:
            accuracy += 1

    return accuracy / min(len(labels), len(predictions))

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape([x_train.shape[0], -1])
x_test = x_test.reshape([x_test.shape[0], -1])


x_train_unnormolized,x_test_unnormalized=x_train,x_test
x_train = x_train.astype(float) / 255.
X_test = x_test.astype(float) / 255.



#MLP with no hidden layer
model_1 = model.MLP(af.ReLU,x_train.shape[1],[],10,0.01)
model_1.fit(x_train,y_train,25,32,True)
prediction_1 = model_1.predict(x_test)
print(evaluate_acc(y_test,prediction_1))


#MLP with a single hidden layer
model_2 = model.MLP(af.ActivationFunction.RELU,x_train.shape[1],[128],10,0.01)
model_2.fit(x_train,y_train,25,32,True)
prediction_2 = model_2.predict(x_test)
print(evaluate_acc(y_test,prediction_2))

#MLP with a twp hidden layers
model_3 = model.MLP(af.ActivationFunction.RELU,x_train.shape[1],[128,128],10,0.01)
model_3.fit(x_train,y_train,25,32,True)
prediction_3 = model_3.predict(x_test)
print(evaluate_acc(y_test,prediction_3))

# plot
x = ['zero', 'one', 'two']
y = [evaluate_acc(y_test, prediction_1), evaluate_acc(y_test, prediction_2), evaluate_acc(y_test, prediction_3)]
plt.bar(x, y, width=0.9, align='center')
i = 1.0
j = -0.5
for i in range(len(x)):
    plt.annotate(y[i], (-0.1 + i, y[i] + j))
plt.title('Fig. 1 Accuracy of these three models')
plt.xlabel('number of hidden layers')
plt.ylabel('accuracy')
plt.show()


#2 layers with sigmoid
model_4 = model.MLP(af.ActivationFunction.SIGMOID,x_train.shape[1],[128,128],10,0.01)
model_4.fit(x_train,y_train,25,32,True)
prediction_4 = model_4.predict(x_test)
print(evaluate_acc(y_test,prediction_4))

#2 layers with tanh
model_5 = model.MLP(af.ActivationFunction.TANH,x_train.shape[1],[128,128],10,0.01)
model_5.fit(x_train,y_train,25,32,True)
prediction_5 = model_5.predict(x_test)
print(evaluate_acc(y_test,prediction_5))

# 2 layers with unnormalized
model_6 = model.MLP(af.ActivationFunction.RELU,x_train.shape[1],[128,128],10,0.01)
model_6.fit(x_train_unnormolized,y_train,25,32,True)
prediction_6 = model_6.predict(x_test_unnormalized)
print(evaluate_acc(y_test,prediction_6))

x = ['sigmoid', 'tanh']
y = [evaluate_acc(y_test, prediction_4), evaluate_acc(y_test, prediction_5)]
plt.bar(x, y, width=0.9, align='center')
i = 1.0
j = -0.5
for i in range(len(x)):
    plt.annotate(y[i], (-0.1 + i, y[i] + j))
plt.title('Fig. 2 Accuracy of models with different activations')
plt.xlabel('activations')
plt.ylabel('accuracy')
plt.show()


def compare_batch_size(x_train,y_train,x_test,y_test):
    batch_sizes = [32,64,128,256,512,1024]
    acc = []
    for size in batch_sizes:
        m = model.MLP(af.ActivationFunction.RELU, x_train.shape[1], [128], 10, 0.01)
        m.fit(x_train, y_train, 25, size, True)
        prediction = m.predict(x_test)
        acc.append(evaluate_acc(y_test, prediction))
        print(size)
    plt.plot(batch_sizes, acc)
    plt.xlabel('Batch Size')
    plt.ylabel('Accuracy')
    plt.title('Accuracy for different Batch Size ')
    plt.show()



def compare_learning_rate(x_train,y_train,x_test,y_test):
    print("comparing learning rate")
    learning_rate = [0.01,0.05,0.1,0.5,1,5,10]
    acc = []
    for rate in learning_rate:
        m = model.MLP(af.ActivationFunction.RELU, x_train.shape[1], [128], 10, rate)
        m.fit(x_train, y_train, 25, 32, True)
        prediction = m.predict(x_test)
        acc.append(evaluate_acc(y_test, prediction))
        print(rate)
    plt.plot(learning_rate, acc)
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.title('Accuracy for different Learning Rate ')
    plt.show()


compare_learning_rate(x_train,y_train,x_test,y_test)