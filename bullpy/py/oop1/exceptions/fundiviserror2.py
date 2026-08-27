def funny_division2 (anumber):
    try:
        if anumber == 13:
            raise ValueError("13 is an unlucky number")
        return 100/ anumber
    except (ZeroDivisionError, TypeError):
        return "enter a number other than zero"

for val in (0, 'helloe', 13, 50):
    print("testing {}:".format(val, end=" "))
    print(funny_division2(val))
