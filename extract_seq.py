from Bio.PDB import *
from QLearning3D import Env
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import math
import numpy as np


def convert_hp(s):
    ret = ""
    for char in s:
        if char in ['A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W', 'G', 'P']:
            ret = ret + 'H'
        else:
            ret = ret + 'P'
    return ret


def plot(seq, positionw):
    x = []
    y = []
    z = []

    fig = plt.figure()
    axes3d = Axes3D(fig)
    for t in range(len(seq)):
        i, j, k = positionw[t]
        x.append(i)
        y.append(j)
        z.append(k)
        if seq[t] == 'H':
            axes3d.scatter(i, j, k, c='b', s=90, zorder=2)
        else:
            axes3d.scatter(i, j, k, c='g', s=90, zorder=2)
    axes3d.plot(x, y, z, linewidth=3, color='black', zorder=1)
    # plt.axis('equal')
    plt.show()


def grid_position(pre, new_pre, cur, spacing):
    dirs = [[spacing, 0, 0], [-spacing, 0, 0], [0, spacing, 0], [0, -spacing, 0], [0, 0, spacing], [0, 0, -spacing]]
    min_dist = float('inf')
    new_cur = None
    for i in range(6):
        dist = (cur[0] - (pre[0] + dirs[i][0])) ** 2 + (cur[1] - (pre[1] + dirs[i][1])) ** 2 + (
                cur[2] - (pre[2] + dirs[i][2])) ** 2
        if dist < min_dist:
            min_dist = dist
            new_cur = [new_pre[0] + dirs[i][0], new_pre[1] + dirs[i][1], new_pre[2] + dirs[i][2]]
    return new_cur


def generate_grid(positions, spacing, start):
    new_positions = positions[:]
    for t in range(len(positions)):
        if t == 0:
            if start:
                new_positions[t] = start
            continue
        cur = positions[t]
        new_cur = grid_position(positions[t - 1], new_positions[t - 1], cur, spacing)
        new_positions[t] = new_cur[:]
    return new_positions


def get_data(file='1fat.pdb', start=None, spacing=1):
    parser = PDBParser()
    structure = parser.get_structure('X', file)
    residues = structure.get_residues()

    ppb = PPBuilder()
    pp = ppb.build_peptides(structure)[0]
    seq = convert_hp(str(pp.get_sequence()))
    calist = pp.get_ca_list()
    positions =[a.get_coord() for a in calist]

    new_positions = generate_grid(positions, spacing, start)
    return seq, new_positions, positions


if __name__ == '__main__':
    seq, new_pos = get_data()
    print(new_pos)
