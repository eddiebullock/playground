#essential  for data processing
score =14
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
else:
    grade = "C"

#list comprehensions (important for ML)
even_number = [x for x in range(10) if x % 2 == 0]
squared = [x**2 for x in range(5)]

#loops 
#processing datasets 
data = [1,2,3,4,5]

#for loops 
for value in data:
    print(value * 2)

#enumerate (useful for indexing)
for index, value in enumerate(data):
    print(f"Index {index}: {value}")

#while loops (less common in ML)
count = 0 
while count < 5:
    print(count)
    count += 1

#functions 
#essential for organizing ML code 
def calculate_mean(numbers):
    return sum(numbers) / len(numbers)

def normalize_data(data):
    """Normalize data to 0-1 range"""
    min_val = min(data)
    max_val = max(data)

    #handle the case where all values are the same 
    if max_val == min_val:
        return [0.0 * len(data)] #or [1.0] * len data depending on the use case
     
    return [(x - min_val) / (max_val - min_val) for x in data]

#example usage
raw_data = [14, 14, 14, 14, 14]
normalized = normalize_data(raw_data)
print(normalized)

