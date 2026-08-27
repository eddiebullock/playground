usernames = ['admin', 'paul', 'max', 'gravy', 'syndrome']

for username in usernames:
    if username == 'admin':
        print('wagwarn drilla')
    else:
        print('welcome ', username)

print('\n')
usernames2 = []
if usernames2:
    for username in usernames2:
        print('wagwarn ', username)
else:
    print('get some users lad')

print('\n')
current_users = ['nan', 'grandad', 'paul', 'ricky', 'donkey']

new_users = ['nan', 'grandad', 'paul', 'nat', 'ant']

current_users_lower = [user.lower() for user in current_users]

for user in new_users:
    if user.lower() in current_users_lower:
        print('username ' + user + ' unavailable')
    else:
        print('username ' + user + ' available')

numbers = list(range(1,10))
for number in numbers:
    if number==1:
        print(number, 'st')
    elif number==2:
        print(number, 'nd')
    elif number==3:
        print(number, 'rd')
    else:
        print(number, 'th')