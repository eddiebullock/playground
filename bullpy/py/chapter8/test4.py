def sandwich(*fillings):
    """summarise whats going in the sandwich"""
    print('the following snadwich will contain: ')
    for filling in fillings:
        print(filling)

sandwich('ham', 'cock', 'balls')
sandwich('ham', 'AH', 'bals')
sandwich('0', '1', '2')

def car_builder(name, make, **details):
    """build car profile"""
    profile = {}
    profile['name'] = name
    profile['make'] = make
    for key, value in details.items():
        profile['key'] = value
    return profile

car = car_builder('paul', 'ford', smell='air freshner', colour='blue')
print(car)