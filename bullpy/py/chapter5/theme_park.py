age = 12

if age < 4:
    print('tickets well cheap')
elif age <18:
    print('ticks aint too bad')
else:
    print('tix well spenny')

if age < 4:
    price = 10
elif age <18:
    price = 20
else:
    price = 20000000

print('for you its, £' + str(price) + ' for entry')