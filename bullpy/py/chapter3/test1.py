for value in range(1,21):
    print(value)

"""milly = list(range(1,1000001))
for value in milly:
    print(value)

print(min(milly))
print(max(milly))
print(sum(milly))"""

odd_list = list(range(1,21,2))
print(odd_list)

thrice_list = list(range(3,30,3))
for values in thrice_list:
    print(values)

for value in range(1,11):
    print(value**3)

cubes = [value**3 for value in range(1,10)]
print(cubes)