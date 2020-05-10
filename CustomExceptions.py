class Error(Exception):
    pass

class loadError(Error):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return('Load Error: {0}'.format(self.message))
        else:
            return('Load Error Occured')

class saveError(Error):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return('Save Error: {0}'.format(self.message))
        else:
            return('Save Error Occured')

