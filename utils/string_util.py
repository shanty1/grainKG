
def all_is_empty(*args):
    for str in args:
        if (str !=None) and (not str.isspace()):
            return False
    return True