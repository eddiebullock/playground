class EvenOnly(list):
    def append(self, integer):
        if not isinstance(integer, int):
            raise TypeError("only integers lad")
        if integer % 2:
            raise TypeError("only even numbers can be added")
        super.append(integer)

try:
    no_return()
except:
    print("i caught an exception")
print("excecuted after the exception")


def funny_division(divider):
    try:
        100 / divider 
    except ZeroDivisionError:
        return "divide by 0 is not a good idea"
    
print(funny_division(0.0))
print(funny_division(50.0))
print(funny_division('hello'))