rivers = {
    'thames': 'london',
    'danube': 'buda',
    'nile': 'egypt'
}

for river, place in rivers.items():
    print('the river ' + river + ' is in ' + place)

print('these are all the river names printed')
for river in rivers:
    print(river.title())

for place in rivers.values():
    print(place.title())