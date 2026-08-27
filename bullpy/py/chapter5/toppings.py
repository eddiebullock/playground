"""toppings_request = 'mushrooms'
if toppings_request != 'anchovies':
    print('hold the anchovies!')

topping_request2 = ['shrooms', 'sezeal', 'pork']
if 'shrooms' in topping_request2:
    print('adding shrooms')
if 'mushrooms' in toppings_request:
    print('dude theyre already added')

print('order up')

requested_toppings = ['shrooms', 'sezeal', 'pork']
for topping in requested_toppings:
    print('Adding ' + topping + ' to the pizza')

print('\ngrubs up')
print('\n')

for topping in requested_toppings:
    if topping == 'pork':
        print('we aint got that shite')
    else:
        print('adding ' + topping + ' to pizza')

print('grubs up 2.0')

requested_toppings2 = []

if requested_toppings2:
    for topping in requested_toppings2:
        print('adding ' + topping + ' to pizza')
    print('grub up 3') 
else:
    print('are you sure you wanit plain doe')"""

available_toppings = ['mushrooms', 'olives', 'green peppers',
 'pepperoni', 'pineapple', 'extra cheese']

requested_toppings = ['mushrooms', 'french fries', 'extra cheese']

for topping in requested_toppings:
    if topping in available_toppings:
        print('adding ' + topping + ' to pizza')
    else:
        print('we aint got that shite')

print('grub up')