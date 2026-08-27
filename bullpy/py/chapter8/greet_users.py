def greet_users(names):
    """greet each user in the list"""
    for name in names:
        msg = print('greetings ', name.title())
        print(msg) 

usernames = ['alex', 'max', 'OG']
greet_users(usernames)