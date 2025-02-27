# linear regression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

fig, ((plot_1, plot_2), (plot_3, _)) = plt.subplots(2,2)

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

    plot_2.set_title("Linear regression")
    #plt.show()

def mse_squares(x,y,b):
    plot_3.scatter(x,y,color = 'b', marker = "o", s = 30)
    y_predict = b[0] + b[1]*x # predicted response vector
    #plot_3.plot(x,y_predict, color = "c") # regression line
    plot_3.axline(xy1 = (0, b[0]), slope = b[1], color = 'cyan')

    bbox = plot_3.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    diff = (plot_3.get_xlim()[1] - plot_3.get_xlim()[0]) / (plot_3.get_ylim()[1] - plot_3.get_ylim()[0]) / (bbox.width / bbox.height) # difference between scales
    
    for patch in plot_3.patches:
        patch.remove()
    for i in range(len(x)): # drawing error squares
        if (y[i] < y_predict[i]):
            patch = patches.Rectangle((x[i], y[i]), -abs(y_predict[i] - y[i]) * diff, abs(y_predict[i] - y[i]), color='purple', alpha=0.5)
        else:
            patch = patches.Rectangle((x[i], y_predict[i]), abs(y_predict[i] - y[i]) * diff, abs(y_predict[i] - y[i]), color='purple', alpha=0.5)
        plot_3.add_patch(patch)
    
    plot_3.set_xlabel(choice_x)
    plot_3.set_ylabel(choice_y)

    plot_2.set_title("MSE squares")


data_path = input("Please input the absolute path to the data file:\n")
# data file needed: student_scores.csv
data = pd.read_csv(data_path, delimiter = ',')
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
    print(data.describe())
    #print("X:\n", "number: ", len(X), "mean: ", X.mean(), "max: ", X.max(), "min:", X.min())
    #print("Y:\n", "number: ", len(Y), "mean: ", Y.mean(), "max: ", Y.max(), "min:", Y.min())
    dots_visualization(X,Y)
    b = estimate_coef(X,Y)
    print("Regression coefficient: ",b)
    regression_line(X,Y,b)
    mse_squares(X,Y,b)
    plt.tight_layout()
    plt.show()
    break
print("Finished.")