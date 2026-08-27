"""alien_0 = {'colour': 'green', 'points': 5}
alien_1 = {'colour': 'blue', 'points': 4}
alien_2 = {'colour': 'yellow', 'points': 3}

aliens = [alien_0, alien_1, alien_2]

for alien in aliens:
    print(alien)

"""

aliens = []

for alien_number in range(30):
    new_alien = {'colour': 'green', 'points': 5, 'speed': 'slow'}
    aliens.append(new_alien)

for alien in aliens[:3]:
    if alien['colour'] == 'green':
        alien['colour'] = 'yellow'
        alien['points'] = '10 points'
        alien['speed'] = 'fast'
    elif alien['colour'] == 'yellow':
        alien['colour'] = 'red'
        alien['points'] = 2
        alien['speed'] = 'medium'

# first five aliens 
for alien in aliens[:5]:
    print(alien)

# count toal aliens 
print('total number of aliens is ', str(len(aliens)))