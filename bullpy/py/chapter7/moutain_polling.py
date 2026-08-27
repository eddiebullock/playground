responses = {}

polling_active = True 

while polling_active:
    name = input('what is your name')
    response = input('what mountain would you like to climb')

    responses[name] = response 

    repeat = input('\nwould you like to let someone else answer "yes/no" ')
    if repeat == 'no':
        polling_active = False

print("\n---polling results---")
for name, response in responses.items():
    print(name + " would like to climb " + response)