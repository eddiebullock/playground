def build_person(first_name, last_name, age=''):
    """build dictionary"""
    person = {'first': first_name, 'second': last_name}
    if age:
        person['age'] = age
    return(person)

musician = build_person('jimi', 'hendrix', '27')
print(musician)