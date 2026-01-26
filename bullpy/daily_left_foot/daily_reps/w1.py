""" variables and types """
x = 1            # int
y = 1.6          # float
name = "alice"   # str
flag = True      # bool


""" operators """
# arithmetic: -, *, +, /, //, %, **
# comparison: ==, >, <, >=, <=, !=
# logical: and, or, not


""" control flow """
# if-elif-else
if x > 5:
    print("big")
elif x == 5:
    print("equal")
else:
    print("small")


# for loop
for i in range(5):
    print(i)

# while loop
n = 5
while n > 0:
    print(n)
    n -= 1


# enumerate and zip
names = ["alice", "bob", "charlie"]
ages = [25, 30, 35]

for idx, name in enumerate(names):
    print(idx, name)

for name, age in zip(names, ages):
    print(name, age)


""" data structures """
# lists
lst = [1, 2, 3, 4, 5]
lst.append(6)
lst.remove(3)
first, last = lst[0], lst[-1]
slice_part = lst[1:3]

squares_list = [x**2 for x in lst]   # list comprehension
print(squares_list)


# dictionaries
d = {"a": 1, "b": 2}
d["c"] = 3

for key, val in d.items():
    print(key, val)


# sets
s = {1, 2, 3}
s.add(4)


# tuples (immutable)
t = (1, 2, 3)


""" functions """
def square(x):
    return x ** 2

def normalize(lst, max_val):
    if max_val == 0:
        return [0.0] * len(lst)
    return [x / max_val for x in lst]

result = square(5)
print(result)


""" file I/O & errors """
# read/write text
with open("data.txt", "w") as f:
    f.write("hello world")

with open("data.txt", "r") as f:
    data = f.read()
    print(data)


# read/write json
import json

data = {"x": [1, 2, 3]}
with open("data.json", "w") as f:
    json.dump(data, f)

with open("data.json", "r") as f:
    data = json.load(f)
    print(data)


# try/except
try:
    x = 1 / 0
except ZeroDivisionError:
    print("can't divide by zero")


""" classes and objects """
class Dataset:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

ds = Dataset([1, 2, 3])
print(len(ds), ds[0], ds[1])


""" numpy """
import numpy as np

x_np = np.array([1, 2, 3])
y_np = x_np * 2
z_np = np.dot(x_np, x_np)
print(y_np, z_np)


""" pytorch """
import torch

a = torch.tensor([1, 2, 3])
b = a * 2
c = torch.dot(a, b)
print(b, c)


