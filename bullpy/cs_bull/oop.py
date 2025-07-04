def main(): 
    name = input("Name:")
    house = input("House:")

    print(f"{name} from {house}")

def get_person():
    name = input("Name:")
    house = input("House:")
    return (name, house)

if __name__ == "__main__":
    main()