"""message = input("tell me something: ")
print(message)"""

"""name = input("whats your name: ")
print("hello, " + name + "!")"""

prompt = "\ntell me somethign and i will repeat it back"
prompt += "\nenter 'quit' to end the program "

active = True 
while active:
    message = input(prompt)
    if message=='quit':
        active = False
    else:
        print('\n', message)