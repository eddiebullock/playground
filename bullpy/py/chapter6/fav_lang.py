favourite_languages = {
    'paul': ['py'],
    'your nan': ['ruby', 'c', 'java'],
    'graham': ['french'],
    'your da': ['py', 'ruby']
} 

for name, languages in favourite_languages.items():
    print(name.title() + "'s favourite languages are: ")
    for language in languages:
        print("\t" + language.title())

"""
print('pauls fav lang is ' + favourite_languages['paul'].title() + '.')
print('\n')

for name, language in favourite_languages.items():
    print(name.title() + "'s fav lang is " + language.title() + '.')

for name in favourite_languages.keys():
    print(name.title())

friends = ['phill', 'naomi', 'paul']
for name in favourite_languages:
    print(name.title())

    if name in friends:
        print('hi ' + name + ' i didnt know you fav lang was ' + favourite_languages[name].title())

if 'erin' not in favourite_languages:
    print('take the poll lov')

for name in sorted(favourite_languages):
    print(name.title(), 'thank you for taking the survey')

print('\n the following languages were mentioned:')
for name in set(favourite_languages.values()):
    print(name.title())


print('\n people that should take the poll')
voters = ['george', 'paul', 'your nan', 'your da'] 
for people in voters:
    if people in favourite_languages:
        print('thank you for taking the poll, ', people)
    else:
        print(people.title(), ' could you please take the poll')"""