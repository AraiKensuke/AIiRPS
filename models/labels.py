def get_a_B_aF0_bF0(label):
    if label == 1:
        #  B / (a+1)  
        a_q2 = 2.01
        B_q2 = 0.1
        a_F0 = 0
        b_F0 = 1
    elif label == 2:
        #  B / (a+1)  
        a_q2 = 4.01
        B_q2 = 0.5
        a_F0 = 0
        b_F0 = 1
    elif label == 3:
        #  B / (a+1)  
        a_q2 = 4.01
        B_q2 = 1.
        a_F0 = 0
        b_F0 = 1
    elif label == 4:
        #  B / (a+1)
        a_q2 = 8.01
        B_q2 = 0.5
        a_F0 = 0
        b_F0 = 1
    elif label == 5:
        #  B / (a+1)
        a_q2 = 8.01
        B_q2 = 2.5
        a_F0 = 0
        b_F0 = 1
    elif label == 6:
        #  B / (a+1)
        a_q2 = 8.01
        B_q2 = 6
        a_F0 = 0
        b_F0 = 1
    elif label == 7:
        #  B / (a+1)
        a_q2 = 20.01
        B_q2 = 0.5
        a_F0 = 0
        b_F0 = 1
    elif label == 8:
        #  B / (a+1)
        a_q2 = 15
        B_q2 = 15
        a_F0 = 0
        b_F0 = 1
    elif label == 9:
        #  B / (a+1)
        a_q2 = 50
        B_q2 = 35
        a_F0 = 0
        b_F0 = 1
    elif label == 11:
        #  B / (a+1)
        a_q2 = 8.01
        B_q2 = 5.5
        a_F0 = 0.9
        b_F0 = 1


    return a_q2, B_q2, a_F0, b_F0
    
