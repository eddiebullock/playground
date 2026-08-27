alien_0 = {'colour': 'green', 'points': 5}
""""print(alien_0['colour'])
print(alien_0['points'])

new_points = alien_0['points']
print('you just earned ' + str(new_points) + ' points!')"""

print(alien_0)
alien_0['x_position'] = 0
alien_0['y_position'] = 25
print(alien_0)

alien_1 = {}
alien_1['colour'] = 'blue'
alien_1['points'] = 6
print(alien_1)

print('\nThe aliens colour is ', alien_0['colour'])
alien_0['colour'] = 'purple'
print('\nthe aliens colour is now ', alien_0['colour'])

alien_3 = {'x_position': 0, 'y_position': 25, 'speed': 'medium'}
print('original position ', str(alien_0['x_position']))

# move alien right 
# decide how far depending on speed 
if alien_3['speed'] == 'slow':
    x_increment = 1
elif alien_3['speed'] == 'medium':
    x_increment = 2
else:
    # must be fast
    x_increment = 3

# new position is x position plus increment 
alien_3['x_position'] = alien_3['x_position'] + x_increment 
print('new alien position ', alien_3['x_position'])

alien_4 = {'colour': 'black', 'speed': 'medium', 'points': 5}
print(alien_4)

del alien_4['points']
print(alien_4)
