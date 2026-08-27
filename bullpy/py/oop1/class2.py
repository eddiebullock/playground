"""class methods and instance variables"""

class Employee:

    num_of_emps = 0 
    raise_amount = 1.04

    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.pay = pay
        self.email = first + "." + last + "@company.com"

        Employee.num_of_emps += 1

    def fullname(self):
        return '{}{}.format(self.first, self.last)'

emp_1 = Employee('your', 'nan', 5000)
emp_2 = Employee('your', 'da', 100)

emp_1.raise_amount = 1.05

print(Employee.num_of_emps)