#include "visualizer.h"
using namespace std;


Visualizer::Visualizer(){}



void Visualizer::load_pointcloud(pcl::visualization::PCLVisualizer::Ptr &viewer, int frame_idx) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ> (pcd_list[frame_idx], *cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
    }
    cout << "File: " << pcd_list[frame_idx] << " ,numbers of points: " << cloud->width * cloud->height  << endl;
    viewer->removeAllShapes();
    viewer->removeAllPointClouds();
    // display cloud
    viewer->setBackgroundColor (0, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud, 255, 255, 255);
    viewer->addPointCloud<pcl::PointXYZ> (cloud, single_color, "sample cloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();
}


vector<vector<float>> Visualizer::draw_cube(pcl::visualization::PCLVisualizer::Ptr &viewer, string file, int type) {
    // read label from file
    vector<vector<float>> labels = get_labels(file);
    // display bbox
    string cube;
    float color_r = 0, color_g = 0, color_b = 0;
    if(type == 0) {
        cube = "gt_car_cube";
        color_r = 1;
    }
    else if(type == 1) {
        cube = "pred_old_car_cube";
        color_g = 1;
    }
    else if(type == 2) {
        cube = "pred_new_car_cube";
        color_b = 1;
    }

    for(int i=0;i<labels.size();i++) {
        string cube_id = cube + to_string(i);
        Eigen::AngleAxisf rotation_vector(labels[i][6], Eigen::Vector3f(0, 0, 1));
        // translation (x, y, z), rotation, hwl
        viewer->addCube(Eigen::Vector3f(labels[i][3], labels[i][4], labels[i][5]),
                        Eigen::Quaternionf(rotation_vector), labels[i][2], labels[i][1], labels[i][0], cube_id);
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, cube_id);
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, color_r, color_g, color_b, cube_id);
    }
    
    return labels;
}


void Visualizer::cloud_visualization()
{
    cout << "frame_idx: " << frame_idx << endl;
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer ("3D Viewer"));
    load_pointcloud(viewer, frame_idx);
    vector<vector<float>> gt_labels = draw_cube(viewer, label_list[frame_idx], 0);
    vector<vector<float>> pred_old_labels = draw_cube(viewer, pred_old_list[frame_idx], 1);
    vector<vector<float>> pred_new_labels = draw_cube(viewer, pred_new_list[frame_idx], 2);

    viewer->addText(to_string(int(gt_labels[0][3])), 0, window_height/2, 30,  1, 0, 0, "distance_text");
    viewer->setSize (window_length, window_height);
    viewer->setCameraPosition(gt_labels[0][3], gt_labels[0][4], gt_labels[0][5] + camera_height,  view_x, view_y, view_z, 
                                up_x, up_y, up_z);
    
    while (!viewer->wasStopped ())
    {
        viewer->spin();
    }
    //viewer->close();
    
    if(search_by_distance || search_by_unmatched) {
        f_id++;
        if(f_id < display_idx_storage.size())
            frame_idx = display_idx_storage[f_id];
        else {
            if(search_by_distance)
                cout << "Search by distance is done" << endl;
            else
                cout << "Search by unmatched is done" << endl;
            exit(0);
        }

    }
    else
        frame_idx++;
    cloud_visualization();
}
