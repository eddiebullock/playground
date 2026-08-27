unconfirmed_users = ['alice', 'joff', 'paul']
confirmed_users = []

while unconfirmed_users:
    current_user = unconfirmed_users.pop()

    print('verifying current user: ', current_user)
    confirmed_users.append(current_user)

print('\nthe following users have been confirmed')
for user in confirmed_users:
    print(user.title())