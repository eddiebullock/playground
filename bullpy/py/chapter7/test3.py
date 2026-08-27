"""sandwich_orders = ['a big one', 'a small one', 'pastrami', 'vegan', 'pastrami', 'meaty', 'pastrami']
finished_sandwiches = []

print('the deli has ran out of pastrami btw')
while 'pastrami' in sandwich_orders:
    sandwich_orders.remove('pastrami')

for sandwich in sandwich_orders:
    print(' i made your ' + sandwich + ' sandwich')
    finished_sandwiches.append(sandwich)

for sandwich in finished_sandwiches:
    print("\nthese sandwiches are finished: ", sandwich)
"""

responses = {}

polling_active = True 

while polling_active:
    name = input('whats your name?')
    response = input('whats your dream vacay?')

    responses[name] = response

    repeat = input('\nwould you like to ask someone else (yes/no)')
    if repeat.lower() == 'no':
        polling_active = False

print("\n===polling results===")
for name, response in responses.items():
    print(name + "'s dream vacay is: " + response)