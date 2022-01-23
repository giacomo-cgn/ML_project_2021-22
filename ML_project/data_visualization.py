import matplotlib.pyplot as plt

color_list = ["blue", "red", "green", "cyan", "magenta", "yellow", "black", "white"]
linestyles_list = ["solid", "dashed", "dashdot", "dotted"]


def visualize(points, func_labels, x_y_labels=['x axis', 'y axis'], x_range=[], y_range=[]):
    """
    This function creates a single graph where different functions can be displayed

    :param points: values of the function(s) to be displayed on the graph
    :param func_labels: labels to be visualized in the legend of the graph
    :param x_y_labels: labels of the x and y axis, respectively
    :param x_range: range [min_value, max_value] to be displayed on the x axis
    :param y_range: range [min_value, max_value] to be displayed on the y axis
    """
    fig, ax = plt.subplots()

    for i in range(len(func_labels)):
        p = points[i] if len(func_labels) > 1 else points
        ax.plot(p, label=func_labels[i],
                linestyle=linestyles_list[i if i < len(linestyles_list) else len(linestyles_list)-i],
                color=color_list[i if i < len(color_list) else len(color_list)-i])
        ax.set_xlabel(x_y_labels[0])
        ax.set_ylabel(x_y_labels[1])
        if x_range:
            ax.set_xlim(x_range)
        if y_range:
            ax.set_ylim(y_range)

    ax.legend()
    plt.show(block=True)
    return


def multiplot_visualization(points, func_labels, x_y_labels, titles, layout=0, x_range=[], y_range=[]):
    """
        This function creates a window where multiple graphs, displaying one function each,
        are shown

        :param points: values of the functions to be displayed on each graph
        :param func_labels: labels to be visualized in the legend of the graphs
        :param x_y_labels: labels of the x and y axis, respectively
        :param titles: titles of each graph
        :param layout: layout of the grid of graphs in the window, Default = 0 so they are displayed in a single column
        :param x_range: range [min_value, max_value] to be displayed on the x axis
        :param y_range: range [min_value, max_value] to be displayed on the y axis
        """
    num_graphs = len(titles)

    fig = plt.figure(1)
    # Displays graphs in a grid
    if layout != 0:
        num_col = layout[0]
        num_row = layout[1]
    else: # Displays graph in a single column
        num_col = 1
        num_row = num_graphs

    num_graphs = 0

    for i in range(num_row):
        for j in range(num_col):

            if num_graphs<len(points):
                # creates subplot
                ax = fig.add_subplot(num_row, num_col, num_graphs + 1)

                if x_range:
                    ax.set_xlim(x_range)
                if y_range:
                    ax.set_ylim(y_range)

                # sets points, labels, titles and axis labels
                ax.plot(points[num_graphs], label=func_labels)
                ax.set_title(titles[num_graphs])
                ax.set_xlabel(x_y_labels[0])
                ax.set_ylabel(x_y_labels[1])
                ax.legend()

                num_graphs += 1

    # if layout is vertical adjust dimensions
    if layout == 0:
        plt.subplots_adjust(0.125, 0.1, 0.9, 0.9, 0.6, 0.6)
    plt.show(block=True)
    return
