import numpy as np
import matplotlib.pyplot as plt


def output(pattern, weight):
    sum = 0.0
    for i in range(len(pattern)):
        sum += pattern[i] * weight[i]
    return 1 if sum > 0 else 0


def error(output, label):
    return label - output


def update(weights, err, pattern):
    for i in range(len(weights)):
        weights[i] += 0.001 * float(err) * pattern[i]


def loadFile():
    file = open('simpleperceptron/iris.data', 'r', encoding='UTF-8')
    data = file.read()
    file.close()
    lines = data.split("\n")

    result = ([])

    for line in lines:
        linesSplitted = line.split(",")
        if(len(linesSplitted) == 5):
            result.append(linesSplitted)

    return result


def test(patterns, weights, labels):
    sum = 0
    rate = 0
    for p in range(len(patterns)):
        sum = 0
        for x in range(len(patterns[p])):
            sum += patterns[p][x] * weights[x]
        result = 1 if sum > 0 else 0
        if labels[p] == result:
            rate += 1
    print(float(rate) / len(patterns))


dataset = loadFile()

patterns = ([])
labels = ([])

for pattern in dataset:
    if pattern[-1] != 'Iris-virginica':
        patterns.append([-1.0] + list(map(float, pattern[0:-1])))
        labels.append(0 if pattern[-1] == 'Iris-setosa' else 1)

weights = np.random.rand(5)

for epoch in range(100):
    sumE = 0

    for p in range(len(patterns)):
        e = error(output(patterns[p], weights), labels[p])
        update(weights, e, patterns[p])
        sumE += e**2

    print(str(epoch) + '/100 epoch: err:' + str(sumE))
    test(patterns, weights, labels)
    if sumE == 0:
        break
