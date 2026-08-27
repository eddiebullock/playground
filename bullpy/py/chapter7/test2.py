"""prompt = "\nwhat pizza toppings do you want"
prompt += "\nenter quit when youre finished" 

while True:
    pizza_topping = input(prompt)
    
    if pizza_topping == 'quit':
        break
    else:
        print("\n yes we'll add " + pizza_topping + " to your pizza")
"""

prompt = "\nhow old are you? (or type quit to exit)"

while True:
    age_input = input(prompt)

    if age_input.lower() == 'quit':
        break

    age = int(age_input)
    if age < 3:
        print('£free')
    elif age <= 12:
        print('£10')
    else:
        print('£500000')
