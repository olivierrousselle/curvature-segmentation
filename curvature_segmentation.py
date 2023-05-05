# -*- coding: utf-8 -*-

import numpy
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
import numpy as np
import open3d as o3d
import math
import matplotlib.pyplot as plt
import pyvista as pv
import time
import os


# =======================================================================================
# Useful functions
# =======================================================================================


def plot(point_cloud):
    """ plot point cloud with Open3D """
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(point_cloud)
    o3d.visualization.draw_geometries([pc])


def curvature_segmentation(centers, normals, part):
    """ segmentation based on the curvature threeshold """
    pc_centers = o3d.geometry.PointCloud()
    pc_centers.points = o3d.utility.Vector3dVector(centers)
    kdtree = o3d.geometry.KDTreeFlann(pc_centers)
    def get_curvature(p):
        K = 10
        [_, pointKNNIdx, pointKNNSquaredDistance] = kdtree.search_knn_vector_3d(centers[p], K)
        knn_index = pointKNNIdx[1:]
        knn_points, knn_normal = centers[knn_index], normals[knn_index]
        P1P0 = knn_points - centers[p]
        P1P0_normal = knn_normal - normals[p]
        numerator = P1P0*P1P0_normal
        denominator = P1P0*P1P0
        K = numerator.sum(axis=1)/denominator.sum(axis=1)
        return (min(K)+max(K))/2, min(K)*max(K), min(K), max(K)
    #KH = np.array([get_curvature(p) for p in range(num_cells)])
    KH, KG, K1, K2 = np.zeros(len(centers)), np.zeros(len(centers)), np.zeros(len(centers)), np.zeros(len(centers))
    for p in range(len(centers)):
        KH[p], KG[p], K1[p], K2[p] = get_curvature(p) 
    if part=="low":
        centers_edge = centers[(KH<-0.3)]
    else:
        centers_edge = centers[(KH<-0.5)]
    #plot(centers_edge)
    return centers_edge


def dbscan_process(centers_edge, part):
    """ DBScan clustering process and selection of the biggest clusters """
    pc_edge = o3d.geometry.PointCloud()
    pc_edge.points = o3d.utility.Vector3dVector(centers_edge)
    if part=="low":
        labels = np.array(pc_edge.cluster_dbscan(eps=0.5, min_points=22))
    else:
        labels = np.array(pc_edge.cluster_dbscan(eps=0.5, min_points=30))
    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / max_label)
    pc_edge.colors = o3d.utility.Vector3dVector(colors[:, :3])
    #o3d.visualization.draw_geometries([pc_edge])
    centers_edge = centers_edge[labels>=0] # delete noisy points
    labels = labels[labels>=0]
    label_nb = np.zeros(max_label+1, dtype=int)
    for i in range(len(labels)):
        label_nb[labels[i]] += 1
    labels_selected = []
    for i in range(len(label_nb)):
        if label_nb[i] >= max(label_nb)/15:
            labels_selected.append(i)
    condition = np.zeros(len(labels), dtype=bool)
    for i in range(len(labels)):
        if labels[i] in labels_selected:
            condition[i] = True
    centers_edge_dbscan = centers_edge[condition]
    #plot(centers_edge_dbscan)
    return centers_edge_dbscan

def morphology_process(centers, centers_edge_dbscan):
    pc_centers = o3d.geometry.PointCloud()
    pc_centers.points = o3d.utility.Vector3dVector(centers)
    kdtree = o3d.geometry.KDTreeFlann(pc_centers)
    def dilation(centers_):
        """ Morphology Dilation operation """
        K = 10
        result = centers_
        centers_to_add = []
        for p in range(len(centers_)):
            [_, pointKNNIdx, _] = kdtree.search_knn_vector_3d(centers_[p], K)
            centers_to_add += [centers[i] for i in pointKNNIdx[1:]]
        result = np.concatenate((result, np.array(centers_to_add)))
        return result
    def erosion(centers_):
        K = 10
        result = centers_
        conditions = np.zeros(len(centers_), dtype=bool)
        for p in range(len(centers_)):
            condition = True
            [_, pointKNNIdx, _] = kdtree.search_knn_vector_3d(centers_[p], K)
            i = 0
            while condition and i<len(pointKNNIdx):
                if centers[pointKNNIdx[i]] not in centers:
                    condition = False
                i+=1
            conditions[p] = condition
        return result[conditions]
    centers_morphology = np.unique(dilation(centers_edge_dbscan), axis=0) 
    return centers_morphology



# =======================================================================================
# Main Program
# =======================================================================================

def main_segmentation(dir_files, name, part):
    
    # Loading files of teeth:
    pdata = pv.read(dir_files + name)    
    # Computation of the normal vectors for each cell
    pdata.compute_normals(inplace=True) 
    normals = pdata['Normals']
    num_cells = len(normals)
    # Computation of the center for each cell
    Centers = pdata.cell_centers()
    centers = np.array([Centers.GetPoint(i) for i in range(num_cells)])
    #plot(centers)
    # Computation of the curvature and segmentation
    centers_edge = curvature_segmentation(centers, normals, part)
    # DBScan process
    centers_edge_dbscan = dbscan_process(centers_edge, part)
    # Morphology process
    centers_edge_dbscan_morphology = morphology_process(centers, centers_edge_dbscan)
    plot(centers_edge_dbscan_morphology)

if __name__ == "__main__":
    
    start_time = time.time()

    dir_path = os.path.dirname(os.path.realpath(__file__)) + '/'
    name_file_seg_output = 'example_up_1.stl'
    main_segmentation(dir_path, name_file_seg_output, "up")
    
    print("Time execution segmentation: %s seconds ---" % round(time.time() - start_time)) 
