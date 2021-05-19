import random

def colored(r, g, b, text):
    '''
    Color text following a RGB code
    '''
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)

def random_hex_colors(n):
    '''
    Define a n-size list of hex-colors
    '''
    colorful_list = []
    
    for i in range(0, n):
        random_number = random.randint(0, 16777215)
        hex_number = str(hex(random_number))
        hex_number = '#' + hex_number[2:]
        colorful_list.append(hex_number)
    
    return colorful_list