prompt = 'tell m a city youve visited'
prompt += '\nenter quit to leave'

while True:
    city = input(prompt)

    if city == 'quit':
        break 
    else:
        print('\ni would love to go to ', city.title())
