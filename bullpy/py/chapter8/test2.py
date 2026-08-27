def city_names(city, country):
    """build dict of cities"""
    return f"{city.title()}, {country.title()}"

place = city_names('london', 'uk')
print(place)

def make_album(artist_name, music_album):
    """display album dict"""
    album_dict = {'artist': artist_name, 'album': music_album}
    return album_dict

while True:
    print('\nname an artist and their album')
    print('\npress q to leave')

    artist = input('artist name: ')
    if artist == 'q':
        break

    album = input('album name: ')
    if album == 'q':
        break

    album_dict = make_album(artist, album)
    print(album_dict)




