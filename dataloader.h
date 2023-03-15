#ifndef _DATALOADER_H_
#define _DATALOADER_H_

#include <iostream>
#include <vector>
#include <string>
#include <unordered_set>
#include <fstream>
#include <sstream>
#include <cassert>
#include "yaml-cpp/yaml.h"

class Dataloader {
public:
    // variable
    std::string dataset_dir;
    std::string label_name;
    std::string fused_name;
    std::string pcl_dir_name;
    std::string inference_dir_name1;
    std::string inference_dir_name2;
    bool search_by_distance;
    int max_distance;
    int min_distance;
    bool search_by_name;
    std::string name;
    bool search_by_unmatched;
    bool search_by_false_positive;
    bool is_new_model;

    std::vector<std::string> pcd_list;
    std::vector<std::string> label_list;
    std::vector<std::string> pred_old_list;
    std::vector<std::string> pred_new_list;

    int frame_idx = 0; 
    int f_id = 0;
    std::unordered_map <std::string, int> name_to_idx_map;
    std::vector<int> display_idx_storage;

    int window_length;
    int window_height;
    double camera_height;
    double view_x;
    double view_y;
    double view_z;
    double up_x;
    double up_y;
    double up_z;

    Dataloader();
    void load_config(YAML::Node config);
    void get_display_start_frames();
    void get_display_start_frames(std::vector<int>unmatched_pred_idx);
    void get_pcd_and_label_list(bool is_fused_label);

    std::vector<std::vector<float>> get_labels(std::string label_file);

};

#endif
