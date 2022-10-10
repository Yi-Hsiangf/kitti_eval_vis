#include <iostream>
#include "metric_calculator.h"
#include "visualizer.h"
#include "yaml-cpp/yaml.h"
#include "dataloader.h"
   
using namespace std;


int main() {
    YAML::Node config = YAML::LoadFile("../visualization.yaml");
    Visualizer visualizer;
    visualizer.load_config(config);
    visualizer.get_pcd_and_label_list();


    Metric_calculator metric_cal;
    metric_cal.get_matched_frame_with_distance(visualizer.label_list, visualizer.pred_old_list, visualizer.pred_new_list);
    metric_cal.show_matched_in_distance(true);
    
    // start the process in loop
    if(visualizer.search_by_unmatched)
        visualizer.get_display_start_frames(metric_cal.get_unmatched_pred(true));
    else
        visualizer.get_display_start_frames();
    visualizer.cloud_visualization();

    return (0);
}
   
