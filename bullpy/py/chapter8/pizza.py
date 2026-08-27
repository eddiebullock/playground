def make_pizza(size, *toppings):
    """display pizza we're making"""
    print('making pizza with the following size: ' + str(size) + ' inches')
    print('making a pizza with the following shite: ')
    for topping in toppings:
        print('adding: ', topping)

make_pizza(16, 'shrooms')
make_pizza(50, 'shrooms', '1', '2')
