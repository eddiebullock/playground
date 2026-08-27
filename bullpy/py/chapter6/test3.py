stan = {
    'species': 'dog',
    'owner': 'alfie'
}
samba = {
    'species': 'dog',
    'owner': 'pen'
}
archie = {
    'species': 'horse',
    'owner': 'pen'    
}
ralph = {
    'species': 'duck',
    'owner': 'joff'
}

pets = [stan, samba, ralph, archie]
for pet in pets:
    print('pets species: ' + pet['species'] + " " + 'owned by: ' + pet['owner']) 

favourite_places = {
    'alice': ['st lucia', 'shepards bush', 'chorely wood'],
    'joff': ['shed', 'bed', 'minimal'],
    'richard': ['boat', 'suff', 'workshop']    
}

for name, place in favourite_places.items():
    print(name + "'s " + "favourite place is ", place)

cities = {
    'london': {
        'stink': 'occasionaly',
        'continent': 'europe',
        'smell': 'doggy'
    },
    'nyc': {
        'stink': 'often',
        'continent': 'north america',
        'smell': 'coppery'
    },
    'mexico': {
        'stink': 'rarely',
        'continent': 'south america',
        'smell': 'orangey'
    }
}

for city, city_info in cities.items():
    print("name: ", city.title())
    smell = city_info['stink'] + " " + city_info['smell']
    continent = city_info['continent']

    print('\tthe sitty smells: ', smell)
    print('\tit is located in ', continent)
    