#setup and env 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
print ("set up complete")

#variavles and data types 
#basic types for ML
name = "yourman" #string
age = 14 #integer
height = 1.75 #float
is_student = True #boolean

#data structures
#list
scores = [1,2,3,4,5]
features = ["age", "income", "education"]
#dictionary
person = {"name": "yourman", "age": 14, "height": 1.75, "salary": 100000}

#essential operations 

#list operations
scores.append(9)
scores[0] #first element
scores[-1] #last element
scores[1:4] #slice from 1 to 4

#basic maths 
import math 
mean_score = sum(scores) / len(scores)
max_score = max(scores)
