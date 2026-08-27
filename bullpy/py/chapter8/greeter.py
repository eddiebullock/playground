def greet_user(username):
    """display greeting"""
    print(username + ", hi im jack" + "!")

greet_user('jack')

def format_name(first_name, last_name):
    """display formatted name"""
    full_name = first_name + ' ' + last_name
    return full_name.title()

#infinite loop 
while True:
    print('\nwhats your name?')
    print("\nenter 'q' at any point to quit")
    
    f_name = input('first name:')
    if f_name == 'q':
        break 
    
    l_name = input('last name:')
    if l_name == 'q':
        break 

    formatted_name = format_name(f_name, l_name)
    print('hi ', formatted_name, '!')
