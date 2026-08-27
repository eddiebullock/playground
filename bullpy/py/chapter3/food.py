fav_food = ['pizza', 'chinese', 'fish n chips', 'your nan']
friends_food = fav_food[:]

fav_food.append('gravy')
friends_food.append('penis')

print('my fav food is: ', fav_food)
print('my friends fav food is ', friends_food)

print(' the first three items on the list are: ', fav_food[:3])
print(' three items in the middle of the list are', fav_food[1:4])

friends_food = fav_food[:]
print(friends_food)
friends_food.append('minstrels')
print(friends_food)

print('my fav foods are: ')
for food in fav_food[:4]:
    print(food)

print('my friends fav foods are: ')
for food in friends_food[:4]:
    print(food)