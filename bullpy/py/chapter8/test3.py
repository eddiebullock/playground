magicians=['1', '2', '3']

def show_magicians(name):
    """display magician names"""
    for magician in name:
        print(magician)

def make_great(name):
    """run through magicians and add the great to them"""
    great_magicians=[]

    while name:
        current_magician = name.pop()
        full_name = current_magician + ' the great'
        great_magicians.append(full_name)

    return great_magicians

great_magicians=make_great(magicians[:])
print('og mag:')
show_magicians(magicians)
print('great mags: ')
show_magicians(great_magicians)