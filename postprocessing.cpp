#include <assert.h>
#include <onnxruntime_c_api.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <map>
#include <fstream>
#include <sys/time.h>
#include <iostream>
#include <numeric>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/video/tracking.hpp"
/*****************************************
* Includes
******************************************/
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <getopt.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <errno.h>
#include <jpeglib.h>
#include <termios.h>
#include <math.h>
#include <iomanip>
#include <sstream>
//#include <CommandAllocatorRing.h>

#include "define.h"
#include "image.h"
//#include "constants.h"

using namespace cv; 
using namespace std;

/*****************************************
* Macros definition
******************************************/
#define RESNET_str "resnet50"
#define MOBILENET_str "mobilenet"
#define NUM_KEYPOINTS 17
#define LOCAL_MAXIMUM_RADIUS 1
#define MAX_POSE_DETECTIONS 10
#define MIN_POSE_SCORE 0.25

/*****************************************
* Global Variables
******************************************/
int model=RESNET50;
std::string model_name = RESNET_str;
char* output_file;
char* input_file;
int stride = 16;
int quant_bytes = 4;
float score_threshold = 0.5;
std::map<int,std::string> label_file_map;
// ONNX Runtime variables
OrtEnv* env;
OrtSession* session;
OrtSessionOptions* session_options;
size_t num_input_nodes;
size_t num_output_nodes;
OrtStatus* status;
//std::vector<const char*> input_node_names(num_input_nodes);
//std::vector<const char*> output_node_names(num_output_nodes);
std::vector<const char*> input_node_names(1);
std::vector<const char*> output_node_names(4);
std::vector<int64_t> input_node_dims_input;
std::vector<int64_t> input_node_dims_shape;
std::vector<int64_t> output_node_dims;
std::vector<OrtValue* > input_tensor(input_node_names.size());

const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

void CheckStatus(OrtStatus* status)
{
    printf("CheckStatus\n");
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "%s\n", msg);
        g_ort->ReleaseStatus(status);
        exit(1);
    }
}

/*****************************************
* Function Name :  loadLabelFile
* Description       : Load txt file
* Arguments         :
* Return value  :
******************************************/
int loadLabelFile(std::string label_file_name)
{
    int counter = 0;
    std::ifstream infile(label_file_name);

    if (!infile.is_open())
    {
        perror("error while opening file");
        return -1;
    }
    std::string line;
    while(std::getline(infile,line))
    {
        label_file_map[counter++] = line;
        //printf("line = %s \n",  label_file_map[counter-1].c_str());
    }
    if (infile.bad())
    {
        perror("error while reading file");
        return -1;
    }
    return 0;
}


int main(int argc, char* argv[])
{
    //output from model and preprocessing
    int image_size = 513;
    int stride = 16;
    
    int arr_size;
    arr_size = ((image_size - 1) / stride) + 1;

    std::string part_names_file("part_names.txt");
    if(loadLabelFile(part_names_file) != 0)
    {
        fprintf(stderr,"Fail to open or process file %s\n",part_names_file.c_str());
        return -1;
    }



    int ret = 0;
    float* heatmap = NULL;
    float* offset = NULL;
    float* fwd = NULL;
    float* bwd = NULL;
    float num;


    float arr_heatmap[arr_size][arr_size][NUM_KEYPOINTS];
    float arr_offset[arr_size][arr_size][NUM_KEYPOINTS*2];
    float arr_fwd[arr_size][arr_size][32];
    float arr_bwd[arr_size][arr_size][32];
    float arr_temp[arr_size*arr_size*34];
    FILE *fp_heatmap;
    FILE *fp_offset;
    FILE *fp_fwd;
    FILE *fp_bwd;
    fp_heatmap = fopen("output_data/heatmap_result.txt", "r");
    fp_offset = fopen("output_data/offsets_result.txt", "r");
    fp_fwd = fopen("output_data/displacement_fwd_result.txt", "r");
    fp_bwd = fopen("output_data/displacement_bwd_result.txt", "r");
    

    int i=0;
    while (fscanf(fp_heatmap, "%f", &num)!=EOF)
    {
        //printf("Value of n=%f \n", num);
        arr_temp[i] = num;
        i++;
    }

    i=0;
    for (int c = 0; c < NUM_KEYPOINTS; c++)
    {
        for (int b = 0; b < arr_size; b++)
        {
            for (int a = 0; a < arr_size; a++)
            {
                arr_heatmap[a][b][c] = arr_temp[i];
                i++;
                //printf("Value of n=%f \n", arr_heatmap[a][b][c]);
            }
        }
    }


    i=0;
    while (fscanf(fp_offset, "%f", &num)!=EOF)
    {
        arr_temp[i] = num;
        i++;
    }
    i=0;
    for (int c = 0; c < NUM_KEYPOINTS*2; c++)
    {
        for (int b = 0; b < arr_size; b++)
        {
            for (int a = 0; a < arr_size; a++)
            {
                arr_offset[a][b][c] = arr_temp[i];
                i++;
                //printf("Value of n=%f \n", arr_offset[a][b][c]);
            }
        }
    }

    i=0;
    while (fscanf(fp_fwd, "%f", &num)!=EOF)
    {
        arr_temp[i] = num;
        i++;
    }
    i=0;
    for (int c = 0; c < 32; c++)
    {
        for (int b = 0; b < arr_size; b++)
        {
            for (int a = 0; a < arr_size; a++)
            {
                arr_fwd[a][b][c] = arr_temp[i];
                i++;
                //printf("Value of n=%f \n", arr_fwd[a][b][c]);
            }
        }
    }

    i=0;
    while (fscanf(fp_bwd, "%f", &num)!=EOF)
    {
        arr_temp[i] = num;
        i++;
    }
    i=0;
    for (int c = 0; c < 32; c++)
    {
        for (int b = 0; b < arr_size; b++)
        {
            for (int a = 0; a < arr_size; a++)
            {
                arr_bwd[a][b][c] = arr_temp[i];
                i++;
                //printf("Value of n=%f \n", arr_bwd[a][b][c]);
            }
        }
    }

    fclose(fp_heatmap);
    fclose(fp_offset);
    fclose(fp_fwd);
    fclose(fp_bwd);

    int lmd = 2 * LOCAL_MAXIMUM_RADIUS + 1;
    float kp_scores[arr_size][arr_size];
    float max_vals[arr_size][arr_size];
    int max_loc[arr_size][arr_size];
    float parts[arr_size*arr_size*NUM_KEYPOINTS][4];
    float max_heat=0;
    int square_x1;
    int square_x2;
    int square_y1;
    int square_y2;
    int flag = 0;
    int part_index = 0;
    float root_score;
    int root_id;
    float root_coord[2];
    float root_image_coords[2];
    int temp_id;
    int pose_count = 0;
    float instance_keypoint_scores[NUM_KEYPOINTS];
    float instance_keypoint_coords[NUM_KEYPOINTS][2];
    float pose_scores[MAX_POSE_DETECTIONS];
    float pose_keypoint_scores[MAX_POSE_DETECTIONS][NUM_KEYPOINTS];
    float pose_keypoint_coords[MAX_POSE_DETECTIONS][NUM_KEYPOINTS][2];
    int squared_nms_radius;
    int nms_radius = 20;
    int num_parts = 17;
    int num_edges = 16;
    int target_keypoint_id, source_keypoint_id;
    float source_keypoint_indices[2];
    float displaced_point[2];
    float displaced_point_indices[2];
    float score;
    float image_coord[2];
    float pose_score;
    float not_overlapped_scores;
    for (int  c = 0; c < NUM_KEYPOINTS; c++)
    {
        for (int b = 0; b < arr_size; b++)
        {
            for (int a = 0; a < arr_size; a++)
            {
                if(arr_heatmap[a][b][c] < score_threshold) kp_scores[a][b] = 0;
                else kp_scores[a][b] = arr_heatmap[a][b][c];
            }
        }

        for (int b = 0; b < arr_size; b++)
        {
            for (int a = 0; a < arr_size; a++)
            {
                max_heat = kp_scores[a][b];
                square_x1 = a - LOCAL_MAXIMUM_RADIUS;
                square_x2 = a + LOCAL_MAXIMUM_RADIUS;
                square_y1 = b - LOCAL_MAXIMUM_RADIUS;
                square_y2 = b + LOCAL_MAXIMUM_RADIUS;
                if(square_x1 < 0) square_x1 = 0;
                if(square_y1 < 0) square_y1 = 0;
                if(square_x2 >= arr_size) square_x2 = arr_size - 1;
                if(square_y2 >= arr_size) square_y2 = arr_size - 1;

                for(int x = square_x1; x <= square_x2; x++)
                {
                    for(int y = square_y1; y <= square_y2; y++)
                    {
                        if(max_heat < kp_scores[x][y]) max_heat = kp_scores[x][y];
                    }
                }
                max_vals[a][b] = max_heat;
                if((kp_scores[a][b] > 0) && (kp_scores[a][b] == max_vals[a][b])) max_loc[a][b] = 1;
                else max_loc[a][b] = 0;
            }
        }

        for (int b = 0; b < arr_size; b++)
        {
            for (int a = 0; a < arr_size; a++)
            {
                if(max_loc[a][b] == 1)
                {
                    parts[part_index][0] = kp_scores[a][b];
                    parts[part_index][1] = c;   // keypoint_id
                    parts[part_index][2] = b;
                    parts[part_index][3] = a;
                    part_index++;
                }
            }
        }

        //sort part base on score: high -> low
        float part_temp[4];
        bool finish_convert = true;
        while(1)
        {
            for (int i = 0; i < part_index; i++)
            {
                if((i < part_index - 1) && (parts[i][0] < parts[i + 1][0]))
                {
                    part_temp[0] = parts[i+1][0];
                    part_temp[1] = parts[i+1][1];
                    part_temp[2] = parts[i+1][2];
                    part_temp[3] = parts[i+1][3];
                    parts[i+1][0] = parts[i][0];
                    parts[i+1][1] = parts[i][1];
                    parts[i+1][2] = parts[i][2];
                    parts[i+1][3] = parts[i][3];
                    parts[i][0] = part_temp[0];
                    parts[i][1] = part_temp[1];
                    parts[i][2] = part_temp[2];
                    parts[i][3] = part_temp[3];
                    finish_convert = false;
                    break;
                }
                else
                {
                    finish_convert = true;
                }
            }
            if(finish_convert == true) break;
        }
    }

/*
    for (int i = 0; i < part_index; i++)
    {
        printf(" \n");
        printf("Value of n=%f \n",  parts[i][0]); // root_score
        printf("Value of n=%f \n",  parts[i][1]); //root_id
        printf("Value of n=%f \n",  parts[i][2]); // coor y
        printf("Value of n=%f \n",  parts[i][3]); // coor x
    }
*/

    for (int i = 0; i < part_index; i++)
    {
        root_score = parts[i][0];
        root_id = int(parts[i][1]);
        root_coord[1] = parts[i][2];
        root_coord[0] = parts[i][3];
        //if(temp_id == root_id) continue;
        squared_nms_radius = pow(nms_radius,2);
        
        //printf(" \n");
        //printf("Value of n = %f \n",  root_score);
        //printf("Value of n = %d \n",  root_id);
        //printf("Value of n = %f \n",  root_coord[0]);
        //printf("Value of n = %f \n",  root_coord[1]);

        
        root_image_coords[0] = root_coord[0] * float(stride) + arr_offset[int(root_coord[0])][int(root_coord[1])][root_id]; 
        root_image_coords[1] = root_coord[1] * float(stride) + arr_offset[int(root_coord[0])][int(root_coord[1])][root_id + 17]; 

        //printf("Value of n = %f \n",  root_image_coord_x);


        if(pose_count != 0)
        {
            float coord[pose_count][2];
            float coord_square[pose_count];
            bool skip_flag = false;
            for(int pose_id = 0; pose_id < pose_count; pose_id++)
            {
                coord[pose_id][0] = pose_keypoint_coords[pose_id][root_id][0];
                coord[pose_id][1] = pose_keypoint_coords[pose_id][root_id][1];
                coord[pose_id][0] = coord[pose_id][0] - root_image_coords[0];
                coord[pose_id][1] = coord[pose_id][1] - root_image_coords[1];
                coord[pose_id][0] = coord[pose_id][0] * coord[pose_id][0];
                coord[pose_id][1] = coord[pose_id][1] * coord[pose_id][1];
                coord_square[pose_id] = coord[pose_id][0] + coord[pose_id][1];
                if(coord_square[pose_id] <= squared_nms_radius) skip_flag = true;
            }
            if(skip_flag == true) continue;
        }
        num_parts = 17;
        num_edges = 16;

        for(int id = 0; id < NUM_KEYPOINTS; id++)
        {
            instance_keypoint_scores[id] = 0;
        }
        instance_keypoint_scores[root_id] = root_score;
        instance_keypoint_coords[root_id][0] = root_image_coords[0];
        instance_keypoint_coords[root_id][1] = root_image_coords[1];

        for(int edge = num_edges-1; edge >= 0; edge--)
        {
            switch(edge)
            {
                case 15:
                    source_keypoint_id = 16;
                    target_keypoint_id = 14;
                break;
                case 14:
                    source_keypoint_id = 14;
                    target_keypoint_id = 12;
                break;
                case 13:
                    source_keypoint_id = 12;
                    target_keypoint_id = 6;
                break;
                case 12:
                    source_keypoint_id = 10;
                    target_keypoint_id = 8;
                break;
                case 11:
                    source_keypoint_id = 8;
                    target_keypoint_id = 6;
                break;
                case 10:
                    source_keypoint_id = 6;
                    target_keypoint_id = 0;
                break;
                case 9:
                    source_keypoint_id = 15;
                    target_keypoint_id = 13;
                break;
                case 8:
                    source_keypoint_id = 13;
                    target_keypoint_id = 11;
                break;
                case 7:
                    source_keypoint_id = 11;
                    target_keypoint_id = 5;
                break;
                case 6:
                    source_keypoint_id = 9;
                    target_keypoint_id = 7;
                break;
                case 5:
                    source_keypoint_id = 7;
                    target_keypoint_id = 5;
                break;
                case 4:
                    source_keypoint_id = 5;
                    target_keypoint_id = 0;
                break;
                case 3:
                    source_keypoint_id = 4;
                    target_keypoint_id = 2;
                break;
                case 2:
                    source_keypoint_id = 2;
                    target_keypoint_id = 0;
                break;
                case 1:
                    source_keypoint_id = 3;
                    target_keypoint_id = 1;
                break;
                case 0:
                    source_keypoint_id = 1;
                    target_keypoint_id = 0;
                break;

                default:
                break;
            }

            //printf("instance_keypoint_scores[source_keypoint_id]: %f \n",  instance_keypoint_scores[source_keypoint_id]);
            //printf("instance_keypoint_scores[target_keypoint_id]: %f \n",  instance_keypoint_scores[target_keypoint_id]);
            
            if((instance_keypoint_scores[source_keypoint_id] > 0) && (instance_keypoint_scores[target_keypoint_id] == 0))
            {
                source_keypoint_indices[0] = float(int(instance_keypoint_coords[source_keypoint_id][0] / float(stride) + 0.5));
                source_keypoint_indices[1] = float(int(instance_keypoint_coords[source_keypoint_id][1] / float(stride) + 0.5));
                if(source_keypoint_indices[0] < 0) source_keypoint_indices[0] = 0;
                else if(source_keypoint_indices[0] > ( arr_size - 1)) source_keypoint_indices[0] = arr_size - 1;
                if(source_keypoint_indices[1] < 0) source_keypoint_indices[1] = 0;
                else if(source_keypoint_indices[1] > (arr_size - 1)) source_keypoint_indices[1] = arr_size - 1;
                //printf("source_keypoint_indices: [%f %f] \n",  source_keypoint_indices[0],source_keypoint_indices[1]);
                displaced_point[0] = instance_keypoint_coords[source_keypoint_id][0] + arr_bwd[int(source_keypoint_indices[0])][int(source_keypoint_indices[1])][edge];
                displaced_point[1] = instance_keypoint_coords[source_keypoint_id][1] + arr_bwd[int(source_keypoint_indices[0])][int(source_keypoint_indices[1])][edge+16];

                displaced_point_indices[0] = float(int(displaced_point[0] / float(stride) + 0.5));
                displaced_point_indices[1] = float(int(displaced_point[1] / float(stride) + 0.5));
                if(displaced_point_indices[0] < 0) displaced_point_indices[0] = 0;
                else if(displaced_point_indices[0] > (arr_size - 1)) displaced_point_indices[0] = arr_size - 1;
                if(displaced_point_indices[1] < 0) displaced_point_indices[1] = 0;
                else if(displaced_point_indices[1] > (arr_size - 1)) displaced_point_indices[1] = arr_size - 1;
                //printf("displaced_point_indices: [%f %f] \n",  displaced_point_indices[0],displaced_point_indices[1]);
                score = arr_heatmap[int(displaced_point_indices[0])][int(displaced_point_indices[1])][target_keypoint_id];
                image_coord[0] = displaced_point_indices[0] * float(stride) + arr_offset[int(displaced_point_indices[0])][int(displaced_point_indices[1])][target_keypoint_id]; 
                image_coord[1] = displaced_point_indices[1] * float(stride) + arr_offset[int(displaced_point_indices[0])][int(displaced_point_indices[1])][target_keypoint_id + 17];
                //printf("score: %f\n",  score);
                //printf("image_coord: [%f %f] \n",  image_coord[0],image_coord[1]);
                instance_keypoint_scores[target_keypoint_id] = score;
                instance_keypoint_coords[target_keypoint_id][0] = image_coord[0];
                instance_keypoint_coords[target_keypoint_id][1] = image_coord[1];
            }
        }

        for(int edge = 0; edge < num_edges; edge++)
        {
            switch(edge)
            {
                case 15:
                    source_keypoint_id = 14;
                    target_keypoint_id = 16;
                break;
                case 14:
                    source_keypoint_id = 12;
                    target_keypoint_id = 14;
                break;
                case 13:
                    source_keypoint_id = 6;
                    target_keypoint_id = 12;
                break;
                case 12:
                    source_keypoint_id = 8;
                    target_keypoint_id = 10;
                break;
                case 11:
                    source_keypoint_id = 6;
                    target_keypoint_id = 8;
                break;
                case 10:
                    source_keypoint_id = 0;
                    target_keypoint_id = 6;
                break;
                case 9:
                    source_keypoint_id = 13;
                    target_keypoint_id = 15;
                break;
                case 8:
                    source_keypoint_id = 11;
                    target_keypoint_id = 13;
                break;
                case 7:
                    source_keypoint_id = 5;
                    target_keypoint_id = 11;
                break;
                case 6:
                    source_keypoint_id = 7;
                    target_keypoint_id = 9;
                break;
                case 5:
                    source_keypoint_id = 5;
                    target_keypoint_id = 7;
                break;
                case 4:
                    source_keypoint_id = 0;
                    target_keypoint_id = 5;
                break;
                case 3:
                    source_keypoint_id = 2;
                    target_keypoint_id = 4;
                break;
                case 2:
                    source_keypoint_id = 0;
                    target_keypoint_id = 2;
                break;
                case 1:
                    source_keypoint_id = 1;
                    target_keypoint_id = 3;
                break;
                case 0:
                    source_keypoint_id = 0;
                    target_keypoint_id = 1;
                break;

                default:
                break;
            }

            //printf("edge: %d \n",  edge);
            //printf("instance_keypoint_scores[source_keypoint_id]: %f \n",  instance_keypoint_scores[source_keypoint_id]);
            //printf("instance_keypoint_scores[target_keypoint_id]: %f \n",  instance_keypoint_scores[target_keypoint_id]);
            
            if((instance_keypoint_scores[source_keypoint_id] > 0) && (instance_keypoint_scores[target_keypoint_id] == 0))
            {
                source_keypoint_indices[0] = float(int(instance_keypoint_coords[source_keypoint_id][0] / float(stride) + 0.5));
                source_keypoint_indices[1] = float(int(instance_keypoint_coords[source_keypoint_id][1] / float(stride) + 0.5));
                if(source_keypoint_indices[0] < 0) source_keypoint_indices[0] = 0;
                else if(source_keypoint_indices[0] > ( arr_size - 1)) source_keypoint_indices[0] = arr_size - 1;
                if(source_keypoint_indices[1] < 0) source_keypoint_indices[1] = 0;
                else if(source_keypoint_indices[1] > (arr_size - 1)) source_keypoint_indices[1] = arr_size - 1;
                //printf("source_keypoint_indices: [%f %f] \n",  source_keypoint_indices[0],source_keypoint_indices[1]);
                displaced_point[0] = instance_keypoint_coords[source_keypoint_id][0] + arr_fwd[int(source_keypoint_indices[0])][int(source_keypoint_indices[1])][edge];
                displaced_point[1] = instance_keypoint_coords[source_keypoint_id][1] + arr_fwd[int(source_keypoint_indices[0])][int(source_keypoint_indices[1])][edge+16];

                displaced_point_indices[0] = float(int(displaced_point[0] / float(stride) + 0.5));
                displaced_point_indices[1] = float(int(displaced_point[1] / float(stride) + 0.5));
                if(displaced_point_indices[0] < 0) displaced_point_indices[0] = 0;
                else if(displaced_point_indices[0] > (arr_size - 1)) displaced_point_indices[0] = arr_size - 1;
                if(displaced_point_indices[1] < 0) displaced_point_indices[1] = 0;
                else if(displaced_point_indices[1] > (arr_size - 1)) displaced_point_indices[1] = arr_size - 1;
                //printf("displaced_point_indices: [%f %f] \n",  displaced_point_indices[0],displaced_point_indices[1]);
                score = arr_heatmap[int(displaced_point_indices[0])][int(displaced_point_indices[1])][target_keypoint_id];
                image_coord[0] = displaced_point_indices[0] * float(stride) + arr_offset[int(displaced_point_indices[0])][int(displaced_point_indices[1])][target_keypoint_id]; 
                image_coord[1] = displaced_point_indices[1] * float(stride) + arr_offset[int(displaced_point_indices[0])][int(displaced_point_indices[1])][target_keypoint_id + 17];
                //printf("score: %f\n",  score);
                //printf("image_coord: [%f %f] \n",  image_coord[0],image_coord[1]);
                instance_keypoint_scores[target_keypoint_id] = score;
                instance_keypoint_coords[target_keypoint_id][0] = image_coord[0];
                instance_keypoint_coords[target_keypoint_id][1] = image_coord[1];
            }
        }

        not_overlapped_scores = 0;
        if(pose_count != 0)
        {
            float coord_cal[pose_count][NUM_KEYPOINTS][2];
            float coord_square_cal[pose_count][NUM_KEYPOINTS];
            bool bigger = false;
            for(int pose_id = 0; pose_id < pose_count; pose_id++)
            {
                for(int key = 0; key < NUM_KEYPOINTS; key++)
                {
                    coord_cal[pose_id][key][0] = pose_keypoint_coords[pose_id][key][0];
                    coord_cal[pose_id][key][1] = pose_keypoint_coords[pose_id][key][1];
                    coord_cal[pose_id][key][0] -= instance_keypoint_coords[key][0];
                    coord_cal[pose_id][key][1] -= instance_keypoint_coords[key][1];
                    coord_cal[pose_id][key][0] = coord_cal[pose_id][key][0] * coord_cal[pose_id][key][0];
                    coord_cal[pose_id][key][1] = coord_cal[pose_id][key][1] * coord_cal[pose_id][key][1];
                    coord_square_cal[pose_id][key] = coord_cal[pose_id][key][0] + coord_cal[pose_id][key][1];
                }
            }
            for(int key = 0; key < NUM_KEYPOINTS; key++)
            {
                for(int pose_id = 0; pose_id < pose_count; pose_id++)
                {
                    if(coord_square_cal[pose_id][key] > squared_nms_radius) bigger = true;
                    else
                    {
                        bigger = false;
                        break;
                    }
                }
                if(bigger == true) not_overlapped_scores += instance_keypoint_scores[key];
            }
        }
        else
        {
            for(int keypoint_id = 0; keypoint_id < NUM_KEYPOINTS; keypoint_id++)
            {
                not_overlapped_scores += instance_keypoint_scores[keypoint_id];
            }
        }
        pose_score = not_overlapped_scores / NUM_KEYPOINTS;

        if(pose_score >= MIN_POSE_SCORE)
        {
            pose_scores[pose_count] = pose_score;
            for(int keypoint_id = 0; keypoint_id < NUM_KEYPOINTS; keypoint_id++)
            {
                pose_keypoint_scores[pose_count][keypoint_id] = instance_keypoint_scores[keypoint_id];
                pose_keypoint_coords[pose_count][keypoint_id][0] = instance_keypoint_coords[keypoint_id][0];
                pose_keypoint_coords[pose_count][keypoint_id][1] = instance_keypoint_coords[keypoint_id][1];
            }
            pose_count += 1;
        }
    }
    for(int pose_id=0; pose_id < pose_count; pose_id++)
    {
        if(pose_scores[pose_id] == 0) break;
        printf("\nPose %d, score = %f \n",  pose_id,pose_scores[pose_id]);
        for(int keypoint_id = 0; keypoint_id < NUM_KEYPOINTS; keypoint_id++)
        {
            printf(" Keypoint = %s \n",  label_file_map[keypoint_id].c_str());
            printf("score = %f, coord = [%f %f]\n",  pose_keypoint_scores[pose_id][keypoint_id], pose_keypoint_coords[pose_id][keypoint_id][0], pose_keypoint_coords[pose_id][keypoint_id][1]);
        }
    }

    return 0;
}
