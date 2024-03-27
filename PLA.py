import numpy as np
import argparse
import os
import time
import matplotlib.pyplot as plt
from Perceptron import LinearPerceptron

def PLA(perceptron: LinearPerceptron) -> np.ndarray:
    st=time.time()
    """
    Do the PLA here on your own.
    weight_matrix -> 3 * 1 resulted weight matrix  

    """
    
    weight_matrix = np.zeros(3)         
    ############START##########
    
    count=0
    data = perceptron.data
    labels = perceptron.label
    
    converged = False  
    
    while not converged:
        converged = True  
        for i in range(len(data)):
            
            x = np.array(data[i])  
            print("this is number:",i,x)
            y = int(labels[i])     
            print(y)
            count += 1
            if np.dot(weight_matrix, x) * y <= 0:  
                
                weight_matrix += y * x  
                converged = False     
                break    
   
    ############END############
    et=time.time()
    execution_time = et - st
    print(st)
    print(et)
    print("Execution time:", execution_time, "seconds")
    print
    print("iterations:",count)
    return weight_matrix

def main(args):
    try:
        if args.path == None or not os.path.exists(args.path):
            raise
    except:
        print("File not found, please try again")
        exit()
    
    
    perceptron = LinearPerceptron(args.path)
    updated_weight = PLA(perceptron=perceptron)
    print("updated weight=" + str(updated_weight))
    #############################################
    #                                           #
    #                                           #
    x_values = np.linspace(min(perceptron.cor_x_pos + perceptron.cor_x_neg), 
                       max(perceptron.cor_x_pos + perceptron.cor_x_neg), 100)

    
    y_values = 3 * x_values + 5
    #weight line
    a=updated_weight[1]
    b=updated_weight[2]
    c=updated_weight[0]
    
    xl=np.linspace(-1000,1000,1000)
    yl = (-a / b) * xl - (c / b)
    plt.plot(xl, yl, label=f'updated_weight:  {a}x + {b}y + {c} = 0',color='black')
    
    # Plot the line
    plt.plot(x_values, y_values, label='baseline:  y = 3x + 5', color='green')
    plt.plot(perceptron.cor_x_pos, perceptron.cor_y_pos, 'bo', label='Positive')
    plt.plot(perceptron.cor_x_neg, perceptron.cor_y_neg, 'ro', label='Negative')
    
    
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Data Points')
    plt.grid(True)
    plt.legend() 
    plt.show()
    #                                           #
    #                                           #
    #############################################

    

if __name__ == '__main__':

    parse = argparse.ArgumentParser(description='Place the .txt file as your path input')
    parse.add_argument('--path', type=str, help='Your file path')
    args = parse.parse_args()
    main(args)