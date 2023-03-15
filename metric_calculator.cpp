#include "metric_calculator.h"

using namespace std;


Metric_calculator::Metric_calculator() {

}


void Metric_calculator::clear_hash_map() {
    gt_distance_to_number_map.clear();
    pred_old_distance_to_number_map.clear();
    pred_new_distance_to_number_map.clear();
}

void Metric_calculator::get_matched_frame_with_distance(vector<string>label_list,vector<string> pred_list) {
    cout << "GT frames: " << label_list.size() << endl;
    cout << "Pred frames: " << pred_list.size() << endl;
    //assert((label_list.size() == pred_list.size()) && "label files not same as pred files");

    clear_hash_map();

    for(int file_idx=0;file_idx<label_list.size();file_idx++) {
        // read labels from file
        vector<vector<float>> gt_labels = get_labels(label_list[file_idx]);
        vector<vector<float>> pred_labels = get_labels(pred_list[file_idx]);

        int matched_counter_new = 0;
        int FP_counter_new = 0;

        for(int gt_idx=0;gt_idx<gt_labels.size();gt_idx++) {
            bool is_matched = false;
            for(int pred_idx=0;pred_idx<pred_labels.size();pred_idx++) {
                float IoU = 0;
                IoU = get_IoU(gt_labels[gt_idx], pred_labels[pred_idx]);
                if(IoU > 0) {
                    is_matched = true;
                    break;
                }
            }
            store_matched_in_distance_map(file_idx, gt_labels[gt_idx][3], is_matched);
            if(is_matched)
                matched_counter_new++;

        }

        FP_counter_new = pred_labels.size() - matched_counter_new;
        store_false_positive(file_idx, FP_counter_new);
    }
}


void Metric_calculator::get_matched_frame_with_distance(vector<string>label_list,vector<string> pred_old_list, vector<string> pred_new_list) {
    cout << "GT frames: " << label_list.size() << endl;
    cout << "Pred old frames: " << pred_old_list.size() << endl;
    cout << "Pred new frames: " << pred_new_list.size() << endl;
    //assert((label_list.size() == pred_list.size()) && "label files not same as pred files");
    
    clear_hash_map();

    for(int file_idx=0;file_idx<label_list.size();file_idx++) {
        // read labels from file
        vector<vector<float>> gt_labels = get_labels(label_list[file_idx]);
        vector<vector<float>> pred_old_labels = get_labels(pred_old_list[file_idx]);
        vector<vector<float>> pred_new_labels = get_labels(pred_new_list[file_idx]);

        int matched_counter_old = 0;
        int matched_counter_new = 0;
        int FP_counter_old = 0;
        int FP_counter_new = 0;

        for(int gt_idx=0;gt_idx<gt_labels.size();gt_idx++) {
            bool is_matched_old = false;
            bool is_matched_new = false;

            for(int pred_idx=0;pred_idx<pred_old_labels.size();pred_idx++) {
                float IoU = 0;
                IoU = get_IoU(gt_labels[gt_idx], pred_old_labels[pred_idx]);
                if(IoU > 0) {
                    is_matched_old = true;
                    break;
                }
            }

            for(int pred_idx=0;pred_idx<pred_new_labels.size();pred_idx++) {
                float IoU = 0;
                IoU = get_IoU(gt_labels[gt_idx], pred_new_labels[pred_idx]);
                if(IoU > 0) {
                    is_matched_new = true;
                    break;
                }
            }
            store_matched_in_distance_map(file_idx, gt_labels[gt_idx][3], is_matched_old, is_matched_new);
            if(is_matched_old)
                matched_counter_old++;
            if(is_matched_new)
                matched_counter_new++;
        }

        FP_counter_new = pred_new_labels.size() - matched_counter_new;
        FP_counter_old = pred_old_labels.size() - matched_counter_old;
        store_false_positive(file_idx, FP_counter_new, FP_counter_old);
    }
}


void Metric_calculator::store_false_positive(int file_idx, int FP_counter_new) {
    if(FP_counter_new > 0)
       false_positive_frames_idx_new.push_back(file_idx);
}


void Metric_calculator::store_false_positive(int file_idx, int FP_counter_new, int FP_counter_old) {
    if(FP_counter_new > 0)
       false_positive_frames_idx_new.push_back(file_idx);
    if(FP_counter_old > 0)
        false_positive_frames_idx_old.push_back(file_idx);
}


void Metric_calculator::store_matched_in_distance_map(int file_idx, float distance, bool is_matched){
        int r_distance = int(distance);
        int m_distance;
        if(distance > 0) 
            m_distance = r_distance - r_distance % 10; // 69 -> 60
        else
            m_distance = r_distance + abs(r_distance) % 10;  // -79 -> -70
            

        gt_distance_to_number_map[m_distance]++;
        if(is_matched)
            pred_new_distance_to_number_map[m_distance]++;
        else
            unmatched_pred_new.push_back(file_idx);
}


void Metric_calculator::store_matched_in_distance_map(int file_idx, float distance, bool is_matched_old, bool is_matched_new){
        int r_distance = int(distance);
        int m_distance;
        if(distance > 0) 
            m_distance = r_distance - r_distance % 10; // 69 -> 60
        else
            m_distance = r_distance + abs(r_distance) % 10;  // -79 -> -70
            

        gt_distance_to_number_map[m_distance]++;
        if(is_matched_old)
            pred_old_distance_to_number_map[m_distance]++;
        else
            unmatched_pred_old.push_back(file_idx);

        if(is_matched_new)
            pred_new_distance_to_number_map[m_distance]++;       
        else
            unmatched_pred_new.push_back(file_idx);
}


vector<int> Metric_calculator::get_unmatched_pred_frames(bool is_new_model) {
    if(is_new_model)
        return unmatched_pred_new;
    else
        return unmatched_pred_old;
}


vector<int> Metric_calculator::get_false_positive_frames(bool is_new_model) {
    if(is_new_model)
        return false_positive_frames_idx_new;
    else
        return false_positive_frames_idx_old;
}



float Metric_calculator::get_IoU(vector<float> gt_label_hwl, vector<float> pred_label_hwl) {
    // hwlxyz
    vector<float> gt_label_xyz = hwlxyz2xyzxyz(gt_label_hwl);
    vector<float> pred_label_xyz = hwlxyz2xyzxyz(pred_label_hwl);
    float IoU = IoU_calculation(gt_label_xyz, pred_label_xyz);

    return IoU;
}

vector<float> Metric_calculator::hwlxyz2xyzxyz(vector<float> bounding_box) {
    float x_min, x_max, y_min, y_max, z_min, z_max, ry; 
    /*
    cout << "bounding box: " << bounding_box[0] << " " <<  bounding_box[1] << " " << bounding_box[2] << 
    " " << bounding_box[3] << " " << bounding_box[4] << " " <<
    bounding_box[5] <<  " " << bounding_box[6] << endl;; 
    */
    x_min = bounding_box[3] - bounding_box[2] / 2;
    x_max = bounding_box[3] + bounding_box[2] / 2;
    
    y_min = bounding_box[4] - bounding_box[1] / 2;
    y_max = bounding_box[4] + bounding_box[1] / 2;
    
    z_min = bounding_box[5] - bounding_box[0] / 2;
    z_max = bounding_box[5] + bounding_box[0] / 2;

    ry = bounding_box[6];
    vector<float> transformed_box {x_min, y_min, z_min, x_max, y_max, z_max}; 
    return transformed_box;
}


float Metric_calculator::IoU_calculation(vector<float> gt_box, vector<float> pred_box) {
    float gt_volume = (gt_box[3]-gt_box[0])*(gt_box[4]-gt_box[1])*(gt_box[5]-gt_box[2]);
    float pred_volume = (pred_box[3]-pred_box[0])*(pred_box[4]-pred_box[1])*(pred_box[5]-pred_box[2]);
    float sum_volume = gt_volume + pred_volume;

    float x1 = max(pred_box[0], gt_box[0]);
    float y1 = max(pred_box[1], gt_box[1]);
    float z1 = max(pred_box[2], gt_box[2]);
    float x2 = min(pred_box[3], gt_box[3]);
    float y2 = min(pred_box[4], gt_box[4]);
    float z2 = min(pred_box[5], gt_box[5]);

    if (x1 >= x2 || y1 >= y2 || z1 >= z2)
        return 0;
    else {
        float intersection_volumne = (x2-x1)*(y2-y1)*(z2-z1);
        return intersection_volumne / (sum_volume-intersection_volumne);
    }
}


void  Metric_calculator::show_metrics(bool is_compared_two_model) {
    if(!is_compared_two_model) 
        cout << "1st model have : " << false_positive_frames_idx_new.size() << endl;
    else {
        cout << "1st model have : " << false_positive_frames_idx_new.size() << " False positive" << endl;    
        cout << "2nd model have : " << false_positive_frames_idx_old.size() << " False positive" << endl;  
    }
    cout << "RCNN distance matched in distance:" << endl;
    for(int distance=-90;distance<=-10;distance+=10) {
        if(!is_compared_two_model)
            cout << "  distance from " << distance << " to " << distance - 10 << ": " 
            << pred_new_distance_to_number_map[distance] << "/" << gt_distance_to_number_map[distance] << endl;
        else
            cout << "  distance from " << distance << " to " << distance - 10 << " in 1st model: " 
            << pred_new_distance_to_number_map[distance] << "/" << gt_distance_to_number_map[distance] << " in 2nd model: " 
            << pred_old_distance_to_number_map[distance] << "/" << gt_distance_to_number_map[distance] << endl;

    }

    if(!is_compared_two_model)
        cout << "  distance from -10 to 10: " << pred_new_distance_to_number_map[0] << "/" << gt_distance_to_number_map[0] << endl;
    else
        cout << "  distance from -10 to 10 in 1st model: " << pred_new_distance_to_number_map[0] << "/" << gt_distance_to_number_map[0] << 
        " in 2nd model: " << pred_old_distance_to_number_map[0] << "/" << gt_distance_to_number_map[0] << endl;

    for(int distance=10;distance<260;distance+=10) {
        if(!is_compared_two_model)
            cout << "  distance from " << distance << " to " << distance + 10 << ": " 
            << pred_new_distance_to_number_map[distance] << "/" << gt_distance_to_number_map[distance] << endl;
        else
            cout << "  distance from " << distance << " to " << distance + 10 << " in 1st model: " 
            << pred_new_distance_to_number_map[distance] << "/" << gt_distance_to_number_map[distance] << " in 2nd model: " 
            << pred_old_distance_to_number_map[distance] << "/" << gt_distance_to_number_map[distance] << endl;
    }

}


