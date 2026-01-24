# list
numbers = list(range(1, 100))

# function to check if even 
def is_even(n):
    return n % 2 == 0 

# print only odd numbers 
print("odd ones:")
for n in numbers:
    if not is_even(n):
        print(n, end=' ')
print("\n")

print("index + value:")
for index, value in enumerate(numbers):
    print(f"{index}: {value}", end=' ')
print("\n")
