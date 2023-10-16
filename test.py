class Car:
    def __init__(self,one,two):
        self.one=one
        self.two=two


    def add(x):
        pass

class Child(Car):
    def __init__(self,one,two):
        self.one=one
        self.two=two
    
    def subtract(x):
        pass
    
    super().__init__(one,two)
    self.one=one
    self.two=two


