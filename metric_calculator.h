#ifndef _METRIC_CALCULATOR_H_
#define _METRIC_CALCULATOR_H_

#include <iostream>
#include <vector>
#include <cassert>
#include <unordered_map>
#include <math.h> 
#include "dataloader.h"

class Metric_calculator : public Dataloader{
private:
    std::unordered_map<int, int> gt_distance_to_number_map;
    std::unordered_map<int, int> pred_old_distance_to_number_map;
    std::unordered_map<int, int> pred_new_distance_to_number_map;
    std::vector<int> unmatched_pred_old;
    std::vector<int> unmatched_pred_new;

    std::vector<int> false_positive_frames_idx_old;
    std::vector<int> false_positive_frames_idx_new;

public:
    Metric_calculator();
    void clear_hash_map();
    void get_matched_frame_with_distance(std::vector<std::string>label_list,std::vector<std::string> pred_list);
    void get_matched_frame_with_distance(std::vector<std::string>label_list,std::vector<std::string> pred_old_list, std::vector<std::string> pred_new_list);
    void store_false_positive(int file_idx, int FP_counter_new);
    void store_false_positive(int file_idx, int FP_counter_new, int FP_counter_old);
    void store_matched_in_distance_map(int file_idx, float distance, bool matched);
    void store_matched_in_distance_map(int file_idx, float distance, bool matched_old, bool matched_new);
    std::vector<int> get_unmatched_pred_frames(bool is_new_model); 
    std::vector<int> get_false_positive_frames(bool is_new_model); 
    float get_IoU(std::vector<float> gt_label_hwl, std::vector<float> pred_label_hwl);
    std::vector<float> hwlxyz2xyzxyz(std::vector<float> bounding_box);
    float IoU_calculation(std::vector<float> gt_box, std::vector<float> pred_box);
    void show_metrics(bool is_compared_two_model);
};

#endif