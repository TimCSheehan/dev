
def check_input_value(x):
    
    if x > 0:
        x = x + 10
        print('adding 10 to input, new number is:', x)
    elif x < 0:
        x = x - 10
        print('subtracting 10 from input, new number is:', x)
    else:
        x = NaN
        print('enter a number')
        
    return x