import sys
import numpy as np
import matplotlib.cm
import matplotlib.pyplot as plt
import gtsam

def plot_graph(graph, values, num_good=None, outlier_marg=None):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    frame = values.atPose3(0)
    for k in range(graph.size()):
        color = 'b'
        alpha = 0.95
        factor = graph.at(k)
        keys = factor.keys()
        # print(keys)
        t1 = frame.transformTo(values.atPose3(keys[0]).translation())
        t2 = frame.transformTo(values.atPose3(keys[1]).translation())
        points = np.array([[t1[0], t1[1], t1[2]],
                           [t2[0], t2[1], t2[2]]])
        ax.plot(points[:,0], points[:,1], points[:,2], color=color, linewidth=0.05, alpha=alpha)
    # Sometimes matplotlib really grinds my gears
    # See: https://github.com/matplotlib/matplotlib/issues/17172#issuecomment-830139107
    ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
    ax.set_axis_off()
    plt.savefig('graph.png', dpi=600)
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} [.g2o file]")
        sys.exit()

    graph, values = gtsam.readG2o(sys.argv[1], True)
    num_good = None
    if len(sys.argv) > 2:
        num_good = np.loadtxt(sys.argv[2])
        print(num_good)

    if len(sys.argv) > 3:
        outlier_marg = np.loadtxt(sys.argv[3])
    else:
        outlier_marg = None

    plot_graph(graph, values, num_good, outlier_marg)

