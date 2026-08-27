pizza = {
    'crust': 'thick',
    'toppings': ['shrooms', 'meat', 'pineapple']
}

print('you ordered a ' + pizza['crust'] + "-crust pizza, with the following toppings:")
for topping in pizza['toppings']:
    print("\t", topping.title())