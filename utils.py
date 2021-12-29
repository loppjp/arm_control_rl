import datetime

def get_datefmt_str():

    n = datetime.datetime.now()

    return n.strftime('%d%m%Y_%H%M%S')