import sys
import numpy as np

# We will use GTSAM to read and write g2o files
import gtsam

# SE-Sync setup
sesync_lib_path = "/Users/kevin/repos/SESync/C++/build/lib"
sys.path.insert(0, sesync_lib_path)

import PySESync

def sesync_pose_to_gtsam_pose(Rmat, tvec, is3D=True):
    if is3D:
        rot = gtsam.Rot3(Rmat)
        tran = gtsam.Point3(tvec)
        pose = gtsam.Pose3(rot, tran)
    else:
        rot = gtsam.Rot2(Rmat)
        tran = gtsam.Point2(tvec)
        pose = gtsam.Pose2(rot, tran)
    return pose

def read_solve_write(base_path, dataset_name, output_path):
    # Staple together filename
    filename = f"{base_path}/{dataset_name}.g2o"

    # Parse g2o file for SESync format
    measurements, num_poses = PySESync.read_g2o_file(filename)
    d = measurements[0].R.shape[0]

    print("Loaded %d measurements between %d %d-dimensional poses from file %s" % (len(measurements), num_poses, d, filename))

    # Set up SESync with 4 threads, use default (Chordal) initialization
    # strategy
    opts = PySESync.SESyncOpts()
    opts.num_threads = 4
    opts.verbose=True

    # Run SESync and get the result
    result = PySESync.SESync(measurements, opts)


    # Extract translational states from solution xhat
    xhat = result.xhat
    R = xhat[:,num_poses:]
    t = xhat[:,0:num_poses]

    # Copy g2o file
    graph, initial = gtsam.readG2o(filename, is3D=(d==3))

    # Make a Values container for GTSAM
    global_opt = gtsam.Values()
    for i in range(num_poses):
        Ri = R[:,d*i:d*(i+1)]
        ti = t[:,i]
        pose = sesync_pose_to_gtsam_pose(Ri, ti, is3D=(d==3))
        global_opt.insert(i, pose)

    gtsam.writeG2o(graph, global_opt, f"{output_path}/{dataset_name}.g2o")

    return

if __name__ == '__main__':
    base_path = "../data/"
    output_path = "../output/"
    datasets = ["cubicle", "parking-garage", "grid3D", "smallGrid3D", "sphere2500",
                "sphere_bignoise_vertex3", "tinyGrid3D", "torus3D"]

    for dataset in datasets:
        read_solve_write(base_path, dataset, output_path)




