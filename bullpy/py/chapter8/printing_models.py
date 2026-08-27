def print_designs(unprinted_designs, current_design):
    """simulate printing designs and add to printed design"""
    while unprinted_designs:
        current_design=unprinted_designs.pop()
        print('printing ' + current_design + ' design')
        printed_designs.append(current_design)

def show_printed_designs(printed_designs):
    print('the following models have been printed: ')
    for designs in printed_designs:
        print(designs)

unprinted_designs=['rombus', 'square', 'circle']
printed_designs=[]

print_designs(unprinted_designs, printed_designs)
show_printed_designs(printed_designs)
print_designs(unprinted_designs[:], printed_designs)