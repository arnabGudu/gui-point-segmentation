import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

x_prev, y_prev = None, None
slope, intercept = None, None
line, text = None, None

def line_eqn(x, y, m, c):
    return y - (m * x + c)

def read_csv(path):
    data = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append([float(i) for i in row])
    return data

def write_csv(path, data):
    with open(path, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(data)

def on_mouse_press(event):
    global x_prev, y_prev
    if event.inaxes is not None:
        x_prev, y_prev = event.xdata, event.ydata

def on_mouse_release(event):
    global slope, intercept, line, text
    if event.inaxes is not None:
        if line:
            line.pop(0).remove()
            text.remove()
        line = plt.plot([x_prev, event.xdata], [y_prev, event.ydata], c='blue')
        slope = (event.ydata - y_prev) / (event.xdata - x_prev)
        intercept = y_prev - slope * x_prev
        text = plt.text(event.xdata, event.ydata, str(f'm={slope:.2f}\nc={intercept:.2f}'), fontsize=12, color='red')
        plt.draw()

def seg(x, y, m, c):
    p = np.zeros(len(x))
    for i in range(len(x)):
        if line_eqn(x[i], y[i], m, c) > 0:
            p[i] = 1
        else:
            p[i] = 0

    X = np.column_stack((x, y))
    model = LogisticRegression()
    model.fit(X, p)
    predictions = model.predict(X)

    coef, intercept = model.coef_[0], model.intercept_[0]
    pm, pc = -coef[0]/coef[1], -intercept/coef[1]
    print(f"y = {pm} x + {pc}")

    plt.scatter(x, y, c=p)
    plt.plot(x, m * x + c, 'r')
    plt.plot(x, pc + pm * x, 'g')
    x_min, _ = plt.gca().get_xlim()
    _, y_max = plt.gca().get_ylim()
    plt.text(x_min, y_max, 'Press q to continue', fontsize=12, color='red')
    plt.show()
    
    return x, y, predictions, pm, pc

def main():
    global slope, intercept
    parser = argparse.ArgumentParser('This script is used to segment the data into 3 classes. \n'
                                     'Please provide the path to the csv file where first column is x, \n'
                                     'second column is y and third column is z (identification index). \n')
    parser.add_argument('path', nargs='?', type=str, default='volume.csv', help='path of the csv file')
    args = parser.parse_args()

    data = read_csv(args.path)
    data = np.array(data).astype(np.float32)
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    plt.gcf().canvas.mpl_connect('button_press_event', on_mouse_press)
    plt.gcf().canvas.mpl_connect('button_release_event', on_mouse_release)
    plt.scatter(x, y, c=z)
    x_min, _ = plt.gca().get_xlim()
    _, y_max = plt.gca().get_ylim()
    plt.text(x_min, y_max, 'Please draw a line to separate the data into 2 classes\nPress q to continue', fontsize=12, color='red')
    plt.show()

    x, y, p, pm, pc = seg(x=x, y=y, m=slope, c=intercept)
    slope, intercept = None, None

    for cl in set(p):
        x1 = np.array([x[i] for i in range(len(x)) if p[i] == cl]).astype(np.float32)
        y1 = np.array([y[i] for i in range(len(x)) if p[i] == cl]).astype(np.float32)
        z1 = np.array([z[i] for i in range(len(x)) if p[i] == cl]).astype(np.float32)

        plt.gcf().canvas.mpl_connect('button_press_event', on_mouse_press)
        plt.gcf().canvas.mpl_connect('button_release_event', on_mouse_release)
        plt.scatter(x1, y1, c=z1)
        x_min, _ = plt.gca().get_xlim()
        _, y_max = plt.gca().get_ylim()
        plt.text(x_min, y_max, 'Please draw a line to separate the data into 2 classes\nPress q to continue', fontsize=12, color='red')
        plt.show()
        
        if slope and intercept:
            break

    x1, y1, p1, pm1, pc1 = seg(x=x1, y=y1, m=slope, c=intercept)

    data = []
    for i in range(len(x)):
        if line_eqn(x[i], y[i], pm, pc) > 0 and line_eqn(x[i], y[i], pm1, pc1) > 0:
            data.append([x[i], y[i], z[i], 2])
        elif line_eqn(x[i], y[i], pm, pc) < 0 and line_eqn(x[i], y[i], pm1, pc1) < 0:
            data.append([x[i], y[i], z[i], 0])
        else:
            data.append([x[i], y[i], z[i], 1])
    write_csv('output.csv', np.array(data))

    plt.scatter(x, y, c=z)
    plt.plot(x, pc + pm * x, 'r')
    plt.plot(x, pc1 + pm1 * x, 'g')
    plt.show()

if __name__ == '__main__':
    main()
