def formatted_name(first_name, last_name, middle=''):
    """display full name"""
    if middle:
        full_name = first_name + ' ' + middle + ' ' + last_name
    else:
        full_name = first_name + ' ' + last_name
    return full_name.title()

magician = formatted_name('david', 'dinamo', 'blane')
print(magician)

magician = formatted_name('barney', 'stinson')
print(magician)