import argparse
import imghdr
import os
import numpy as np
import cv2
import open3d as o3d

def process_images(input_folder, output_file):
    # Get the image files from the input folder
    image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if imghdr.what(os.path.join(input_folder, f)) in ['jpeg', 'png', 'bmp']]

    # Load the images
    images = [cv2.imread(image_file) for image_file in image_files]

    # Estimate the depth maps for each image
    depth_maps = [cv2.estimate_depth(image) for image in images]

    # Convert the depth maps to point clouds
    point_clouds = [o3d.geometry.PointCloud(depth_map) for depth_map in depth_maps]

    # Merge the point clouds into a single point cloud
    point_cloud = o3d.geometry.PointCloud()
    for pc in point_clouds:
        point_cloud += pc

    # Downsample the point cloud
    point_cloud = point_cloud.voxel_down_sample(voxel_size=0.001)

    # Create a mesh from the point cloud
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = point_cloud.points
    mesh.triangles = point_cloud.triangle

    # Save the mesh to an OBJ file
    o3d.io.write_triangle_mesh(output_file, mesh)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='Path to the input folder containing the images')
    parser.add_argument('-o', '--output', required=True, help='Path to the output OBJ file')
    args = parser.parse_args()

    process_images(args.input, args.output)

