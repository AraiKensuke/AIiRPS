def get_a_B(label):
    if label == 1:
        #  B / (a+1)  
        a_q2 = 2.01
        B_q2 = 0.1
    elif label == 2:
        #  B / (a+1)  
        a_q2 = 4.01
        B_q2 = 0.5
    elif label == 3:
        #  B / (a+1)  
        a_q2 = 4.01
        B_q2 = 1.
    elif label == 4:
        #  B / (a+1)
        a_q2 = 8.01
        B_q2 = 0.5
    elif label == 5:
        #  B / (a+1)
        a_q2 = 8.01
        B_q2 = 2.5

    return a_q2, B_q2
    
