#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>

#include <thread>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <pcl/visualization/cloud_viewer.h>


#include "yaml-cpp/yaml.h"
   
using namespace std;


class Visualizer {
private:
    string dataset_dir;
    string txt_file;
    bool search_by_distance;
    int max_distance;
    int min_distance;
    bool search_by_name;
    string name;
    vector<string> pcd_list;
    vector<string> label_list;
    vector<string> pred_old_list;
    vector<string> pred_new_list;

    int frame_idx = 0; 
    int f_id = 0;
    unordered_map <string, int> name_to_idx_map;
    vector<int> frame_idx_in_distance;

public:
    Visualizer(){

    }

    void load_config(YAML::Node config) {
        dataset_dir = config["dataset_dir"].as<string>();
        txt_file = config["txt_file"].as<string>();
        search_by_distance = config["search_by_distance"].as<bool>();
        max_distance = config["max_distance"].as<int>();
        min_distance = config["min_distance"].as<int>();
        search_by_name = config["search_by_name"].as<bool>();
        name = config["name"].as<string>();
    }

    
    void get_display_start_frames() {
        assert((!search_by_name || !search_by_distance) && "search_by_name and search_by_distance both True");
    
        if(search_by_name) {
            frame_idx = name_to_idx_map[name];
        }
        else if(search_by_distance) {
            for(int frame_idx=0;frame_idx<label_list.size();frame_idx++) {
                vector<vector<float>> labels = get_labels(label_list[frame_idx]);
                for(int j=0;j<labels.size();j++) {
                    if(min_distance < labels[j][3] && labels[j][3] < max_distance) {
                        frame_idx_in_distance.push_back(frame_idx);
                        continue;
                    }
                }
            }
            frame_idx = frame_idx_in_distance[f_id]; // f_id = 0
        }
    }


    void get_pcd_and_label_list() {
        string textfile = dataset_dir + txt_file;
        std::ifstream in_file(textfile);
        string line;
        while (getline(in_file, line)) {
            string pcd_file = dataset_dir + "/pcl/" + line + ".pcd";
            string label_file = dataset_dir + "/label/" + line + ".txt";
            string pred_old_file = dataset_dir + "/pred_old/" + line + ".txt";
            string pred_new_file = dataset_dir + "/pred_new/" + line + ".txt";
            
            name_to_idx_map[line] = pcd_list.size();
            pcd_list.push_back(pcd_file);
            label_list.push_back(label_file);
            pred_old_list.push_back(pred_old_file);
            pred_new_list.push_back(pred_new_file);
        }
    }


    vector<vector<float>> get_labels(string label_file) {
        vector<vector<float>> labels;
        string line;
        ifstream file (label_file);
        if (file.is_open())
        {
            while(getline(file,line))
            {
                vector<float> label;
                stringstream ss(line);
                string token;
                int counter = 0;
                while (!ss.eof()) {
                    getline(ss, token, ' ');
                    if(counter >= 8 && counter <= 14)
                        label.push_back(std::stof(token));
                    counter++;
                }
                labels.push_back(label);
            }
            file.close();
        }
        else 
            cout << "Unable to open file: " << label_file << endl;
        return labels;
    }

   void load_pointcloud(pcl::visualization::PCLVisualizer::Ptr &viewer, int frame_idx) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
        if (pcl::io::loadPCDFile<pcl::PointXYZ> (pcd_list[frame_idx], *cloud) == -1) //* load the file
        {
            PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
        }
        cout << "Load pointcloud, data points: " << cloud->width * cloud->height  << endl;
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


   vector<vector<float>> draw_cube(pcl::visualization::PCLVisualizer::Ptr &viewer, string file, int type) {
        // read label from file
        vector<vector<float>> labels = get_labels(file);
        // display bbox
        string cube;
        float color_r = 0, color_g = 0, color_b = 0;
        if(type == 0) {
            cube = "gt_car";
            color_r = 1;
        }
        else if(type == 1) {
            cube = "pred_old_car";
            color_g = 1;
        }
        else if(type == 2) {
            cube = "pred_new_car";
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


    void cloud_visualization()
    {
        pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer ("3D Viewer: " + label_list[frame_idx]));
        load_pointcloud(viewer, frame_idx);
        vector<vector<float>> gt_labels = draw_cube(viewer, label_list[frame_idx], 0);
        vector<vector<float>> pred_old_labels = draw_cube(viewer, pred_old_list[frame_idx], 1);
        vector<vector<float>> pred_new_labels = draw_cube(viewer, pred_new_list[frame_idx], 2);

        viewer->setCameraPosition(gt_labels[0][3], gt_labels[0][4], gt_labels[0][5] + 30, 0,-1,-1);

        while (!viewer->wasStopped ())
        {
            viewer->spin();
        }
        //viewer->close();
        
        if(search_by_distance) {
            f_id++;
            frame_idx = frame_idx_in_distance[f_id];
        }
        else
            frame_idx++;
        cloud_visualization();
    }

};




int main() {
    YAML::Node config = YAML::LoadFile("../visualization.yaml");
    Visualizer visualizer;
    visualizer.load_config(config);
    visualizer.get_pcd_and_label_list();
    visualizer.get_display_start_frames();
    visualizer.cloud_visualization();
    return (0);
}
   
