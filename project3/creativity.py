from keras.datasets import mnist
from sklearn.neural_network import MLPClassifier


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


x_train = x_train.astype(float) / 255.
X_test = x_test.astype(float) / 255.


clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=10, batch_size=32,alpha=0,
                    learning_rate_init=0.01).fit(x_train, y_train)
predicted = clf.predict(x_test)

print(evaluate_acc(y_test,predicted))
