
scores = [1,2,3,4,5,6,7,8,9,10]

mean_score = sum(scores) / len(scores)
max_score = (max(scores))
min_score = (min(scores))

#find numbers above av
above_average = [num for num in scores if num > mean_score]

print(mean_score)
print(max_score)
print(min_score)
print(above_average)

