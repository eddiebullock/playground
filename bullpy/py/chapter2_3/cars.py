cars = ['toyota', 'ford', 'vw', 'opal']
cars.sort()
print(cars)
cars.sort(reverse=True)
print(cars)

print('here is the original list: ')
print(cars)

print('here is the sorted list: ')
print(sorted(cars))

print('here is the original list again: ')
print(cars)
cars.reverse()
print(cars)

no_cars = len(cars)
print("No. cars: ", no_cars)