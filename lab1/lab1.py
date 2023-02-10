print("---Exercice 1.4.1---")

sizeOfTriangle = 5

triangle = []

for i in range(sizeOfTriangle):
    triangle.append((i+1) * "x")

for j in range(len(triangle) - 1, 0, -1):
    triangle.append(j * "x")

for line in triangle:
    print(line)

print("---Exercice 1.4.2---")

input_str = "n45as29@#8ss6"

#I'm appending a litteral character at the end of the string to force an exception to be raised when casting to an int for the last int
input_str += "q"

result = 0
number = "0"

for char in input_str:
    try:
        int(char)
        number += char
    except:
        result += int(number)
        number = "0"
print("The sum of numbers of the input string is:" + str(result))


print("---Exercice 1.4.3---")

integer = 42
division = 0
bit = 0
binary = ""

not_complete = True

while not_complete:
    bit = integer % 2
    result = integer // 2
    binary = str(bit) + binary
    integer = result
    if(result == 0):
        not_complete = False

print("42 in binary is: " + binary)

print("---Exerice 1.5.1---")

def getNFibo(n: int) -> int:
    if(n == 0 or n == 1):
        return n
    else:
        return getNFibo(n-1) + getNFibo(n-2)

def fibonaci(upper_threshold: int) -> list:
    array = []
    i = 0
    result = 0
    loop = True
    while loop:
        result = getNFibo(i)
        if(result < upper_threshold):
            array.append(result)
        else:
            loop = False
        i += 1
    return array

print(fibonaci(10))

print("---Exercice 1.5.2---")

def display_as_digi(number: float) -> None:
    numbers = {
        '0':['xxx', 'x x', 'x x', 'x x', 'xxx'],
        '1':['  x','  x','  x','  x'],
        '2':['xxx','  x', 'xxx','x  ','xxx'],
        '3':['xxx', '  x', 'xxx', '  x','xxx'],
        '4':['x x', 'x x', 'xxx', '  x', '  x'],
        '5':['xxx', 'x  ', 'xxx', '  x', 'xxx'],
        '6':['xxx', 'x  ', 'xxx', 'x x', 'xxx'],
        '7':['xxx', '  x', '  x', '  x', '  x'],
        '8':['xxx', 'x x', 'xxx', 'x x', 'xxx'],
        '9':['xxx', 'x x', 'xxx', '  x', '  x'],
        ".":['   ', '   ', '   ', '   ', ' x ']
    }

    for line in range(5):
        str_line = ""
        for char in str(number):
            str_line += numbers[char][line] + " "
        print(str_line)

display_as_digi(42.69)

print("--Exercice 2---")

import numpy as np
import matplotlib.pyplot as plt
import time

def matrix_treshold_slow(matrix, lower_treshold: int):
    start = time.time()

    for index, x in np.ndenumerate(matrix):
        if x < lower_treshold:
            matrix[index] = 0

    print("Time elapsed :" + str(time.time() - start))

def matrix_treshold(matrix, lower_treshold):
    start = time.time()

    matrix[matrix < lower_treshold] = 0

    print("Time elapsed (without for):" + str(time.time() - start))
    
matrix = np.arange(25, 0, -1).reshape(5,5)

matrix_treshold(matrix, 10)

matrix2 = np.arange(25, 0, -1).reshape(5, 5)

matrix_treshold_slow(matrix, 10)

#I didn't implement all the digits but this is the same logic process for all.
def show_in_digi(number):
    numbers = {
        0:[[False, False, False], [False, True, False], [False, True, False], [False, True, False], [False, False, False]],
        1:[[True, False, False], [True, True, False], [True, True, False], [True, True, False], [True, True, False]],
        2:[[False, False, False], [True, True, False], [False, False, False], [False, True, True], [False, False, False]]
    }
    black_bar = np.full((5, 1), True)
    display = black_bar
    for char in str(number):
        display = np.hstack((display, numbers[int(char)]))
        display = np.hstack((display, black_bar))
        
    plt.imshow(display, cmap='binary')

show_in_digi(2121)
plt.show()

print("---Exercice 3---")

import pandas as pd

dataset = pd.read_csv("./california_housing_test.csv")

print(dataset.loc[dataset.total_bedrooms > 310, dataset.columns[0:]])

households_mean = dataset.households.mean()

mean = dataset.mean().mean()
dataset.fillna(mean)

fig, (longlat, household) = plt.subplots(2)

household.plot(dataset.index, dataset.households, "ro")
household.axhline(y=households_mean, color='b', linestyle='-')

longlat.plot(dataset.latitude, dataset.longitude, "ro")
plt.show()

dataset.median_income = (dataset.median_income - dataset.median_income.min()) / (dataset.median_income.max() - dataset.median_income.min())
print(dataset.median_income)

dataset.population = (dataset.population - dataset.population.min()) / (dataset.population.max() - dataset.population.min())
print(dataset.population)

print(dataset.corr())
