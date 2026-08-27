"""static, class, regular methods
    regular methods pass isntance first as 'init, class methods pass class (cls) first, static methods dont apss anything automatically (they act as functions)"""


class Employee:

    num_of_emps = 0 
    raise_amt = 1.04

    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.pay = pay
        self.email = first + "." + last + "@company.com"

        Employee.num_of_emps += 1

    def fullname(self):
        return '{}{}.format(self.first, self.last)'
    
    @classmethod
    def set_raise_amt(cls, amount):
        cls.raise_amt = amount

    @classmethod
    def from_string(cls, emp_str):
        first, last, pay = emp_str_1.split('-')
        return cls(first, last, pay)
    
    @staticmethod
    def is_workday(day):
        if day.weekday() == 5 or day.weekday == 6:
            return False
        else:
            return True 
        
emp_1 = Employee('your', 'nan', 5000)
emp_2 = Employee('your', 'da', 100)

import datetime
my_date = datetime.date(2026, 8, 11)

print(Employee.is_workday(my_date))


# emp_str_1 = 'your-nan-5000'
# emp_str_2 = 'your-da-56000'
# emp_str_3 = 'your-bro-57000'

# new_emp_1 = Employee.from_string(emp_str_1)

# print(new_emp_1.pay)
# print(new_emp_1.email)