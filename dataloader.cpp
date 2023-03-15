#include "dataloader.h"

using namespace std;

Dataloader::Dataloader() {}


void Dataloader::load_config(YAML::Node config) {
    dataset_dir = config["dataset_dir"].as<string>();
    label_name = config["label_name"].as<string>();
    fused_name = config["fused_name"].as<string>();
    pcl_dir_name = config["pcl_dir_name"].as<string>();
    inference_dir_name1 = config["inference_dir_name1"].as<string>();
    inference_dir_name2 = config["inference_dir_name2"].as<string>();
    search_by_distance = config["search_by_distance"].as<bool>();
    max_distance = config["max_distance"].as<int>();
    min_distance = config["min_distance"].as<int>();
    search_by_name = config["search_by_name"].as<bool>();
    name = config["name"].as<string>();
    search_by_unmatched = config["search_by_unmatched"].as<bool>();
    search_by_false_positive = config["search_by_false_positive"].as<bool>();   
    is_new_model = config["is_new_model"].as<bool>();

    camera_height = config["camera_height"].as<float>();
    view_x = config["view_x"].as<float>();
    view_y = config["view_y"].as<float>();
    view_z = config["view_z"].as<float>();
    up_x = config["up_x"].as<float>();
    up_y = config["up_y"].as<float>();
    up_z = config["up_z"].as<float>();

    window_length = config["window_length"].as<int>();
    window_height = config["window_height"].as<int>();

    assert((!search_by_name || !search_by_distance || !search_by_unmatched || !search_by_false_positive) && "ERROR searching query");
}


void Dataloader::get_display_start_frames() {
    if(search_by_name) {
        frame_idx = name_to_idx_map[name];
    }
    else if(search_by_distance) {
        for(int frame_idx=0;frame_idx<label_list.size();frame_idx++) {
            vector<vector<float>> labels = get_labels(label_list[frame_idx]);
            for(int j=0;j<labels.size();j++) {
                if(min_distance <= labels[j][3] && labels[j][3] <= max_distance) {
                    display_idx_storage.push_back(frame_idx);
                    continue;
                }
            }
        }
        cout << "Displaying " << display_idx_storage.size() << " frames" << endl;
        frame_idx = display_idx_storage[f_id]; // f_id = 0
    }
}


void Dataloader::get_display_start_frames(vector<int>query_frames) {
    display_idx_storage = query_frames;
    cout << "Displaying " << display_idx_storage.size() << " frames" << endl;
    frame_idx = display_idx_storage[f_id]; // f_id = 0
}



void Dataloader::get_pcd_and_label_list(bool is_fused_label) {
    string pcd_file, label_file, pred_old_file, pred_new_file;
    if(!is_fused_label) {
        string text_file = dataset_dir + label_name;
        std::ifstream in_file(text_file);
        string line;
        while (getline(in_file, line)) {
            pcd_file = dataset_dir + pcl_dir_name + line + ".pcd";
            label_file = dataset_dir + "/label/" + line + ".txt";
            pred_old_file = dataset_dir + inference_dir_name1 + line + ".txt";
            pred_new_file = dataset_dir + inference_dir_name2 + line + ".txt";
            
            name_to_idx_map[line] = pcd_list.size();
            pcd_list.push_back(pcd_file);
            label_list.push_back(label_file);
            pred_old_list.push_back(pred_old_file);
            pred_new_list.push_back(pred_new_file);
        }
    }
    else {
        // Include fused dataset
        std::unordered_set<int> label_name_set;
        string label_file = dataset_dir + label_name;
        std::ifstream in_label_file(label_file);
        string line;
        while (getline(in_label_file, line)) {
            label_name_set.insert(stoi(line));
        }

        string fused_file = dataset_dir + fused_name;
        std::ifstream in_fused_file(fused_file);
        while (getline(in_fused_file, line)) {
            int id = stoi(line);
            if(label_name_set.count(id) > 0) {
                pcd_file = dataset_dir + pcl_dir_name + line + ".pcd";
                label_file = dataset_dir + "/label/" + line + ".txt";
                pred_old_file = dataset_dir + inference_dir_name1 + line + ".txt";
                pred_new_file = dataset_dir + inference_dir_name2 + line + ".txt";
            }
            else {
                int remain = id % 5;
                if(remain >= 3)
                    id = id + 5 - (remain);
                else
                    id = id - remain; 
                string label_id = to_string(id);
                label_id = string(6 - label_id.size(), '0') + label_id;
                pcd_file = dataset_dir + "/pcl/" + line + ".pcd";

                label_file = dataset_dir + "/label/" + label_id + ".txt";
                pred_old_file = dataset_dir + inference_dir_name1 + label_id + ".txt";
                pred_new_file = dataset_dir + inference_dir_name2 + line + ".txt";
            }
            name_to_idx_map[line] = pcd_list.size();
            pcd_list.push_back(pcd_file);
            label_list.push_back(label_file);
            pred_old_list.push_back(pred_old_file);
            pred_new_list.push_back(pred_new_file);
        }
    }
}


vector<vector<float>> Dataloader::get_labels(string label_file) {
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


