
class Parameters:
    def __repr__(self):
        return str(self.__dict__)

    def __copy__(self):
        newone = type(self)()
        newone.__dict__.update(self.__dict__)
        return newone

    def copy(self):
        newone = type(self)()
        newone.__dict__.update(self.__dict__)
        return newone