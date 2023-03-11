#include "dataloader.h"

using namespace std;

Dataloader::Dataloader() {}


void Dataloader::load_config(YAML::Node config) {
    dataset_dir = config["dataset_dir"].as<string>();
    txt_file = config["txt_file"].as<string>();
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



void Dataloader::get_pcd_and_label_list() {
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


