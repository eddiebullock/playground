name = 'Paul'
age = 5000

if age >= 50:
    print('you an alien lad?')
else:
    print('you young')

if name.lower() == 'paul':
    print('true')
else:
    print('nah lad')

if age > 100 and name == 'Paul':
    print('do you know simon peg?')
else:
    print('who are ya?')

aliens = ['paul', 'ET', 'my parents']
if 'paul' in aliens:
    print('found him lads')
else:
    print('still lost')

# actual way 
print(name == 'paul')
print(name.lower == 'paul')
print(age > 42)
print(age == 5000)

print('paul' in aliens)
print('barry' in aliens)
print('barry' not in aliens)