from ucimlrepo import fetch_ucirepo 
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

all_data = fetch_ucirepo(id = 186).data

length = int(len(all_data.targets) * 0.8) 

x = np.array(all_data.features[:length])
y = np.array(all_data.targets[:length])

x_test = np.array(all_data.features[length:])
y_test = np.array(all_data.targets[length:])

weights = np.random.rand(11, 1)
bias = 0
learning_rate = 0.000114

def compute_quality(input): # Returns prediction for a single X value
    result = np.dot(input, weights) + bias
    return result

def MSE(datay, predictions): # Mean square error with all predictions
    return np.sum((datay - predictions) ** 2) * 1/len(datay)

def gradient_descent(datay, datax, predictions): # With all predictions
    dw = -np.matmul((datay - predictions).T, datax)
    return (np.sum(dw, axis=0, keepdims=True) * 1/len(datax))

def adjust_weights(gradient): #Adjust weights with gradient calculation
    global weights
    
    weights -= gradient * learning_rate

myepochs = []
mymse = []

EPOCHS = 5000
for i in range(EPOCHS):
    predictions = []
    print("Epoch", i)
    myepochs.append(i)
    
    for i, wine in enumerate(x):
        prediction =  compute_quality(wine)
        predictions.append(prediction)

    gradient = gradient_descent(y, x, np.array(predictions))
    adjust_weights(gradient.T)
    
    mse = MSE(y, predictions)
    mymse.append(mse)
    print("Loss: ", mse)
    print("*"*30)

print('FINAL WEIGHTS')
print(weights)

plt.plot(myepochs[1000:], mymse[1000:])

plt.xlabel('Loss')
plt.ylabel('Epoch')
plt.title('LOSS curve')

plt.show()

print("TESTS")
print("*"*50)

predictions_test = []
indices = []

for i, wine in enumerate(x_test):
    indices.append(i)
    prediccion = compute_quality(x_test)
    predictions_test.append(prediccion)
    
print("All tests loss: ", MSE(y_test, np.array(predictions_test)))

plt.scatter(indices, y_test, color="red")
plt.scatter(indices, np.squeeze(predictions_test)[0])
plt.show()