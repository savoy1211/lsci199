class Parent:
    def __init__(self, born_year, country_origin):
        self.born_year = born_year
        self.country_origin = country_origin
        self.age = self.init_age()
    
    def init_age(self):
        return 2020 - self.born_year
    
class Child(Parent):
    def __init__(self, born_year, country_origin):
        self.born_year = born_year
        self.country_origin = country_origin
        self.age = self.init_age()