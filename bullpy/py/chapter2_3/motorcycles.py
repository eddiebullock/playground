motorcycles = ['honda', 'yamahah', 'suzuki']
print(motorcycles)

motorcycles[0] = 'ducati'
print(motorcycles)

motorcycles.append('ducati')
print(motorcycles)

motorcycles = []
motorcycles.append('ducati')
motorcycles.append('honda')
motorcycles.append('your nan')
print(motorcycles)

motorcycles = ['yamahah', 'honda', 'yournan']
motorcycles.insert(0, 'yourdad')
print(motorcycles)

motorcycles = ['yamahah', 'honda', 'yournan']
del motorcycles[0]
print(motorcycles)

motorcycles = ['yamahah', 'honda', 'yournan']
popped_motorcycles = motorcycles.pop()
print(motorcycles)
print(popped_motorcycles)

motorcycles = ['yamahah', 'honda', 'yournan']
last_motorcycle  = motorcycles.pop()
print("the last motorcycle i owned was " + last_motorcycle.title() + ".")
first_owned = motorcycles.pop(0)
print("the first motorcycle i owned was a " + first_owned.title() + ".")

#removing stuff 
motorcycles = ['yamahah', 'honda', 'yournan']
motorcycles.remove('yournan')
print(motorcycles)

motorcycles = ['yamahah', 'honda', 'yournan']
ugly = 'yournan'
motorcycles.remove(ugly)
print(motorcycles)
print("\nA " + ugly.title() + " is too ugly for me.")

motorcycles = ['yamahah', 'honda', 'yournan']
print(motorcycles[-1])