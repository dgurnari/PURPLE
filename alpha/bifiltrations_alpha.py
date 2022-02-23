import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt


# computes each simplex contribution to the ECC
def compute_ECC_contributions_alpha(point_cloud, dbg=False):
    alpha_complex = gd.AlphaComplex(points=point_cloud)
    simplex_tree = alpha_complex.create_simplex_tree()
    
    ecc = {}
    num_simplices = 0
    
    for s, f in simplex_tree.get_filtration():
        dim = len(s) - 1
        ecc[f] = ecc.get(f, 0) + (-1)**dim

        num_simplices += 1
        
        if dbg: 
            print(s, f)
            
    # remove the contributions that are 0
    to_del = []
    for key in ecc:
        if ecc[key] == 0:
            to_del.append(key)
    for key in to_del:
        del ecc[key]
        
    return (sorted(list(ecc.items()), key = lambda x: x[0]),
            num_simplices)

def compute_ECP_contributions_alpha(point_cloud, vertex_filtrations, dbg=False):
    alpha_complex = gd.AlphaComplex(points=point_cloud)
    simplex_tree = alpha_complex.create_simplex_tree()
    
    ecp = {}
    num_simplices = 0
    
    for s, f1 in simplex_tree.get_filtration():
        f2 = max([vertex_filtrations[i] for i in s])
        dim = len(s) - 1

        ecp[(f1, f2)] = ecp.get((f1, f2), 0) + (-1)**dim

        num_simplices += 1
        
        if dbg: 
            print(s, f1, f2)
            
    # remove the contributions that are 0
    to_del = []
    for key in ecp:
        if ecp[key] == 0:
            to_del.append(key)
    for key in to_del:
        del ecp[key]
        
    return (sorted(list(ecp.items()), key = lambda x: x[0]), 
            num_simplices)


# given a list of +-1 contributions, computes the ECC
# TODO: clean the code
def ECC_from_contributions(local_contributions):

    euler_characteristic = []
    old_f, current_characteristic = local_contributions[0]

    for filtration, contribution in local_contributions[1:]:
        if filtration > old_f:
            euler_characteristic.append([old_f, current_characteristic])
            old_f = filtration

        current_characteristic += contribution

    # add last contribution
    if len(local_contributions) > 1:
        euler_characteristic.append([filtration, current_characteristic])
        
    if len(local_contributions) == 1:
        euler_characteristic.append(local_contributions[0])

    return euler_characteristic




def EC_at_filtration(contributions, f):
    return sum([c[1] for c in contributions if (c[0] <= f)])


def EC_at_bifiltration(contributions, f1, f2):
    return sum([c[1] for c in contributions if (c[0][0] <= f1) and
                                               (c[0][1] <= f2)])
    
def subsample_ECC(contributions, f_range, size=51):
    
    ecc = np.zeros(size)
    
    for i, f in enumerate(np.linspace(f_range[0], f_range[1], num=size)):
        ecc[i] = EC_at_filtration(contributions, f)
            
    return ecc

def subsample_ECP(contributions, f1_range, f2_range, size=(51, 51)):
    
    ecp = np.zeros(size)
    
    for i, f1 in enumerate(np.linspace(f1_range[0], f1_range[1], num=size[0])):
        for j, f2 in enumerate(np.linspace(f2_range[0], f2_range[1], num=size[1])):
            ecp[i,j] = EC_at_bifiltration(contributions, f1, f2)
            
    return ecp



# Plotting functions
def plot_ECP(contributions, ranges,
             this_ax=None, 
             colorbar=False, **kwargs):
    
    f1min, f1max, f2min, f2max = ranges
    
    if this_ax == None:
        this_ax = plt.gca()
    
    f1_list = sorted(set( [f1min] + [c[0][0] for c in contributions if (c[0][0] > f1min) & (c[0][0] < f1max)] + [f1max]))
    f2_list = sorted(set( [f2min] + [c[0][1] for c in contributions if (c[0][1] > f2min) & (c[0][1] < f2max)] + [f2max]))
    
    Z = np.zeros((len(f2_list)-1, len(f1_list)-1))

    for i, f1 in enumerate(f1_list[:-1]):
        for j, f2 in enumerate(f2_list[:-1]):
            Z[j,i] = EC_at_bifiltration(contributions, f1, f2)
    
    # Plotting
    im = this_ax.pcolormesh(f1_list, f2_list, Z)

    this_ax.set_xlabel("Filtration 1")
    this_ax.set_ylabel("Filtration 2")
    
    if colorbar:
        plt.colorbar(im, ax=this_ax)
    
    return this_ax

def plot_ECC(e_list, this_ax=None, with_lines=False, **kwargs):

    if this_ax == None:
        this_ax = plt.gca()

    # Plotting
    this_ax.scatter([f[0] for f in e_list], [f[1] for f in e_list])
    # draw horizontal and vertical lines b/w points
    if with_lines:
        for i in range(1, len(e_list)):
            this_ax.vlines(
                x=e_list[i][0],
                ymin=min(e_list[i - 1][1], e_list[i][1]),
                ymax=max(e_list[i - 1][1], e_list[i][1]),
            )
            this_ax.hlines(y=e_list[i - 1][1], xmin=e_list[i - 1][0], xmax=e_list[i][0])

    this_ax.set_xlabel("Filtration")
    this_ax.set_ylabel("Euler Characteristic")
    return this_ax

