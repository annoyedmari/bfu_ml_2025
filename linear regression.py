# linear regression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def estimate_coef(x,y):
    n = np.size(x) # number of points
    m_x = np.mean(x) # mean of x vector
    m_y = np.mean(y) # mean of y vector
    SS_xx = np.sum(x*x) - n*(m_x)*(m_x) # deviation about x
    SS_xy = np.sum(y*x) - n*(m_y)*(m_x) # cross-deviation about x
    # regression coef
    b1 = SS_xy / SS_xx
    b0 = m_y - b1*m_x
    return (b0, b1)

def dots_visualization(x,y):
    plot_1.scatter(x,y,color = "m", marker = "o", s = 30) # plotting points as scatter plot
    plot_1.set_xlabel(choice_x)
    plot_1.set_ylabel(choice_y)
    plot_1.set_title("Dots visualization")
    #plt.show()

def regression_line(x,y,b):
    plot_2.scatter(x,y,color = "m", marker = "o", s = 30) # plotting points as scatter plot
    y_predict = b[0] + b[1]*x # predicted response vector
    plot_2.plot(x,y_predict, color = "g") # regression line

    plot_2.set_xlabel(choice_x)
    plot_2.set_ylabel(choice_y)

    plot_2.set_title("Regression test")
    #plt.show()


fig, ((plot_1, plot_2), (plot_3, _)) = plt.subplots(2,2) # TODO: add plot 3!!
data = pd.read_csv('D:\Telegram Desktop\student_scores.csv', delimiter = ',')
headers = list(data.columns.values)
print(headers)
while True:
    choice_x = input("Which header would you like to use as X vector? (Input a name.)\n")
    if (choice_x not in headers):
        print("Such header does not exist.")
        break
    print("You've chosen the header", choice_x)
    choice_y = input("Which header would you like to use as Y vector? (Input a name.)\n")
    if (choice_y not in headers):
        print("Such header does not exist.")
        break
    elif (choice_x==choice_y):
        print("Cannot select the same column.")
        break
    print("You've chosen the header", choice_y)
    X = data[choice_x].to_numpy()
    Y = data[choice_y].to_numpy()
    print("X:\n", "number: ", len(X), "mean: ", X.mean(), "max: ", X.max(), "min:", X.min())
    print("Y:\n", "number: ", len(Y), "mean: ", Y.mean(), "max: ", Y.max(), "min:", Y.min())
    # TODO: make multiple graphics show at once!!
    dots_visualization(X,Y)
    b = estimate_coef(X,Y)
    print("Regression coefficient: ",b)
    regression_line(X,Y,b)
    plt.tight_layout()
    plt.show()
    # TODO: 3rd graphic with squared of errors (shaded)
    break
print("Finished.")