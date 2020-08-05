/*
Note restriction:
1) Model is used with architecture is MobileNet with output stride is fixed with 16
2) Quantity bytes is 4
*/

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

using namespace cv; 
using namespace std;

/*****************************************
* Macros definition
******************************************/
#define RESNET_str "resnet50"
#define MOBILENET_str "mobilenet"

/*****************************************
* Global Variables
******************************************/
int model=RESNET50;
std::string model_name = RESNET_str;
char* output_file;
char* input_file;
int stride = 16;
int quant_bytes = 4;

// ONNX Runtime variables
OrtEnv* env;
OrtSession* session;
OrtSessionOptions* session_options;
size_t num_input_nodes;
size_t num_output_nodes;
OrtStatus* status;
float* out_data[4];// = NULL;

//std::vector<const char*> input_node_names(num_input_nodes);
//std::vector<const char*> output_node_names(num_output_nodes);
std::vector<const char*> input_node_names(1);
std::vector<const char*> output_node_names(4);
std::vector<int64_t> input_node_dims_input;
std::vector<int64_t> input_node_dims_shape;
std::vector<int64_t> output_node_dims;
std::vector<OrtValue* > input_tensor(input_node_names.size());

const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

struct S_Pixel
{
    unsigned char RGBA[3];
};

void CheckStatus(OrtStatus* status)
{
    //printf("CheckStatus\n");
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "%s\n", msg);
        g_ort->ReleaseStatus(status);
        exit(1);
    }
}


/*****************************************
* Function Name : sigmoid
* Description   : helper function for YOLO Post Processing
* Arguments :
* Return value  :
******************************************/
float sigmoid(float x){
    return 1.0/(1.0+exp(-x));
}

/*****************************************************************************
    FORWARD DECLRATIONS
 *****************************************************************************/
/*****************************************
* Function Name : parse_argument
* Description   :
* Arguments :
* Return value  :
******************************************/
int parse_argument(int argc, char* argv[])
{   
    int ret = 0;
	int index = 1;

	for (index = 1; index < argc; index++) {
		if (!strcmp("-model", argv[index])) {
			model_name = argv[index+1];
		} else if (!strcmp("-stride", argv[index])) {
			stride = atoi(argv[index+1]);
		} else if (!strcmp("-ifile", argv[index])) {
			input_file = argv[index+1];
		} else if (!strcmp("-ofile", argv[index])) {
			output_file = argv[index+1];
		} else if (!strcmp("-quant_bytes", argv[index])) {
			quant_bytes = atoi(argv[index+1]);
		}
        else {
        }
    }

    // process some input argument
    if (!strcmp(model_name.c_str(), RESNET_str)) {
        model = RESNET50;
    }
    else if (!strcmp(model_name.c_str(), MOBILENET_str)) {
        model = MOBILENET;
    }
    else {
        printf("The architecture is not supported\n");
        ret = -1;
    }

    // below process is related to restriction
    if (model == MOBILENET) {
        stride = 16;
        quant_bytes = 4;
    }
    return ret;
}

// valid resolution
void valid_resolution(int width, int height, int *target_width, int *target_height) {
    *target_width = (width / stride) * stride +1;
    *target_height = (height / stride) * stride +1;
}
void prepare_ONNX_Runtime() {
    //ONNX runtime: Necessary
    CheckStatus(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));

    //ONNX runtime: Necessary
    g_ort->CreateSessionOptions(&session_options);
    g_ort->SetInterOpNumThreads(session_options, 2); //Multi-core

    //Config : model
    //std::string onnx_model_name = model_name + "-PoseNet.onnx";
    std::string onnx_model_name = model_name + "-PoseNet.onnx";
    std::string onnx_model_path= "./models/" + onnx_model_name;

    //ONNX runtime load model
    CheckStatus(g_ort->CreateSession(env, onnx_model_path.c_str(), session_options, &session));    
    printf("Start Loading Model %s\n", model_name.c_str());
}

cv::Mat add_means_of_RGB_channel(cv::Mat _img, cv::Scalar image_net_mean) {

    cv::Mat img;

    _img.convertTo(img, CV_32FC3);
   
    //struct S_Pixel
    //{
    //    unsigned char RGBA[3];
    //};
    int img_sizex = img.cols;
    int img_sizey = img.rows;

/* 
    for ( size_t y = 0; y < 5; y++){
        for ( size_t x = 0; x < 5; x++){
            cout << "Before" << endl;
            cout << "img[" << x << "," << y << "][0] = " << img.at<cv::Vec3f>(x,y)[0] << endl;
            cout << "img[" << x << "," << y << "][1] = " << img.at<cv::Vec3f>(x,y)[1] << endl;
            cout << "img[" << x << "," << y << "][2] = " << img.at<cv::Vec3f>(x,y)[2] << endl;
        }
    }
*/ 

// hard code here, check again
    for ( size_t y = 0; y < img_sizey; y++){
        for ( size_t x = 0; x < img_sizex; x++){
            img.at<cv::Vec3f>(x,y)[0] += image_net_mean[0]; //-123.15;
            img.at<cv::Vec3f>(x,y)[1] += image_net_mean[1]; //-115.90;
            img.at<cv::Vec3f>(x,y)[2] += image_net_mean[2]; //-103.06;
        }
    }
/* 
    for ( size_t y = 0; y < 5; y++) {
        for ( size_t x = 0; x < 5; x++) {
            cout << "After" << endl;
            cout << "img[" << x << "," << y << "][0] = " << img.at<cv::Vec3f>(x,y)[0] << endl;
            cout << "img[" << x << "," << y << "][1] = " << img.at<cv::Vec3f>(x,y)[1] << endl;
            cout << "img[" << x << "," << y << "][2] = " << img.at<cv::Vec3f>(x,y)[2] << endl;
        }
    }    
    printf("after plus\n");
*/
    return img;
}

cv::Mat mobilenet_process_input(cv::Mat _img)
{
    cv::Mat img;
    //input_img = input_img * (2.0 / 255.0) - 1.0  // normalize to [-1,1]
    //input_img = input_img.reshape(1, target_height, target_width, 3)  // NHWC

    cv::Scalar image_net_mean(-123.15, -115.90, -103.06);
    img = add_means_of_RGB_channel(_img, image_net_mean);
    img = img/255;

    return img;
}

cv::Mat resnet_process_input(cv::Mat _img)
{
    cv::Mat img;
    //cv::Scalar image_net_mean(-123.15, -115.90, -103.06);
    //img_update = add_means_of_RGB_channel(img, image_net_mean);
    //img_update = img_update/255;
    return img;
}

int preprocess_input() {
    int ret=0;
    const char* mat_out = "mat_out.jpg";
    int tget_wid, tget_hei, in_wid, in_hei, in_channel;
    in_channel = 3;
    int img_sizex, img_sizey, img_channels;

    // process image
    cv::Mat _img = cv::imread(input_file, cv::IMREAD_COLOR);  
    cv::Mat img;
    cv::Mat img_update;   

    valid_resolution(_img.cols, _img.rows, (int*) &tget_wid, (int*) &tget_hei); 
    //this model use fixed 257x257 image with stride = 32
    tget_wid = 257;
    tget_hei = 257;

    cv::resize(_img, img, cv::Size(tget_wid, tget_hei), 0, 0, CV_INTER_LINEAR);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    cv::imwrite(mat_out, img);

    stbi_uc * img_data = stbi_load(mat_out, &img_sizex, &img_sizey, &img_channels, STBI_default);
    const S_Pixel * imgPixels(reinterpret_cast<const S_Pixel *>(img_data));

    size_t input_tensor_size = tget_wid * tget_hei * in_channel;
    std::vector<float> input_tensor_values(input_tensor_size);

    if(model == MOBILENET)
    {
        int offs = 0;
        for (int c = 0; c < 3; c++){
            for (int y = 0; y < img_sizey; y++){
                for (int x = 0; x < img_sizex; x++, offs++){
                    const int val(imgPixels[y * img_sizex + x].RGBA[c]);
                    input_tensor_values[offs] = ((float)val)*2/255 - 1; // for mobilenet
                }
            }
        }
    }
    else if(model == RESNET50)
    {
        //cv::Scalar image_net_mean(-123.15, -115.90, -103.06);
        //img_update = add_means_of_RGB_channel(img, image_net_mean);
        //img_update = img_update/255;
    }
    

    // create input tensor object from data values
    OrtMemoryInfo* memory_info;
    CheckStatus(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));

    // print model input layer (node names, types, shape etc.)    
    OrtAllocator* allocator;
    CheckStatus(g_ort->GetAllocatorWithDefaultOptions(&allocator));
    status = g_ort->SessionGetInputCount(session, &num_input_nodes);
    status = g_ort->SessionGetOutputCount(session, &num_output_nodes);
    printf("\nCurrent Model is %s\n",model_name.c_str());
    printf("Number of inputs = %zu\n", num_input_nodes);
    printf("Number of outputs = %zu\n", num_output_nodes);

    // print input tensor type before setting value 
    for (size_t i = 0; i < num_input_nodes; i++){
        // print input node names
        char* input_name;
        status = g_ort->SessionGetInputName(session, i, allocator, &input_name);
        printf("Input %zu : name=%s\n", i, input_name);
        input_node_names[i] = input_name;

        // print input node types
        OrtTypeInfo* typeinfo;
        status = g_ort->SessionGetInputTypeInfo(session, i, &typeinfo);
        const OrtTensorTypeAndShapeInfo* tensor_info;
        CheckStatus(g_ort->CastTypeInfoToTensorInfo(typeinfo,&tensor_info));
        ONNXTensorElementDataType type;
        CheckStatus(g_ort->GetTensorElementType(tensor_info,&type));
        printf("Input %zu : type=%d\n", i, type);
        size_t num_dims;

        if(i == 0)
        {
            num_dims = 4;
            printf("Input %zu : num_dims=%zu\n", i, num_dims);
            input_node_dims_input.resize(num_dims);
            g_ort->GetDimensions(tensor_info, (int64_t*)input_node_dims_input.data(), num_dims);
            //check input dim
            for (size_t j = 0; j < num_dims; j++) printf("Input %zu : dim %zu=%jd\n", i, j, input_node_dims_input[j]);
            
            //this is fixed model, so skip the update input value step
            //input_node_dims_input[0]=1;
            //input_node_dims_input[1]= tget_wid;
            //input_node_dims_input[2]= tget_hei;
            //input_node_dims_input[3]= in_channel;

            //check input dim
            for (size_t j = 0; j < num_dims; j++) printf("After set value: Input %zu : dim %zu=%jd\n", i, j, input_node_dims_input[j]);
        }
        else  {
            printf("incorrect input tensor dim count is %zu", i);
        }
        

        g_ort->ReleaseTypeInfo(typeinfo);
    }    
    
    for (size_t i = 0; i < num_output_nodes; i++) {
        // print input node names
        char* output_name;
        CheckStatus(g_ort->GetAllocatorWithDefaultOptions(&allocator));
        CheckStatus(g_ort->SessionGetOutputName(session, i, allocator, &output_name));
        //printf("output %d : name=%s\n", i, output_name);
        output_node_names[i] = output_name;
        // print input node types
        OrtTypeInfo* typeinfo;
        CheckStatus(g_ort->SessionGetOutputTypeInfo(session, i, &typeinfo));
        const OrtTensorTypeAndShapeInfo* tensor_info;
        CheckStatus(g_ort->CastTypeInfoToTensorInfo(typeinfo,&tensor_info));
        ONNXTensorElementDataType type;
        CheckStatus(g_ort->GetTensorElementType(tensor_info,&type));
        printf("Output : output_name=%s\n",  output_node_names[i]);
        printf("Output %d : type=%d\n", i, type);
        // print input shapes/dims
        size_t num_dims = 4;
        printf("Output %d : num_dims=%zu\n", i, num_dims);
        output_node_dims.resize(num_dims);
        g_ort->GetDimensions(tensor_info, (int64_t*)output_node_dims.data(), num_dims);
        for (size_t j = 0; j < num_dims; j++) printf("output %zu : dim %zu=%jd\n", i, j, output_node_dims[j]);
        g_ort->ReleaseTypeInfo(typeinfo);
    }

    // set value for input tensor
    CheckStatus(g_ort->CreateTensorWithDataAsOrtValue(memory_info, input_tensor_values.data(), input_tensor_size*sizeof(float), input_node_dims_input.data(), 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor[0]));
    int is_tensor;
    CheckStatus(g_ort->IsTensor(input_tensor[0],&is_tensor));
    assert(is_tensor);
    g_ort->ReleaseMemoryInfo(memory_info);
    return ret;
}

int postprocess() {
    int ret = 0;

    // check output tensor
    for(int i=0; i< 20; i++)
    {
        printf("output 0: %f\n", sigmoid(out_data[0][i]));
    }
    for(int i=0; i< 20; i++)
    {
        printf("output 1: %f\n", out_data[1][i]);
    }
    for(int i=0; i< 20; i++)
    {
        printf("output 2: %f\n", out_data[2][i]);
    }
    for(int i=0; i< 20; i++)
    {
        printf("output 3: %f\n", out_data[3][i]);
    }

    int image_size = 257;
    int stride = 32;
    int arr_size = ((image_size - 1) / stride) + 1;



    return ret;
}

void run_model(){
    // RUN: score model & input tensor, get back output tensor
    std::vector<OrtValue *> output_tensor(4);
    output_tensor[0] = NULL;
    output_tensor[1] = NULL;
    output_tensor[2] = NULL;
    output_tensor[3] = NULL;
    int is_tensor;

    // check parameter
    CheckStatus(g_ort->IsTensor(input_tensor[0], &is_tensor));
    assert(is_tensor);

    CheckStatus(g_ort->Run(session, NULL, input_node_names.data(), input_tensor.data(), num_input_nodes, output_node_names.data(), num_output_nodes, output_tensor.data()));

    for (int i = 0; i <= 3; i++)
    {
        CheckStatus(g_ort->IsTensor(output_tensor[i],&is_tensor));
        assert(is_tensor);
        
        // Get pointer to output tensor float values
        g_ort->GetTensorMutableData(output_tensor[i], (void**)&out_data[i]);        
    }
}

/*****************************************
* Function Name : timedifference_msec
* Description   :
* Arguments :
* Return value  :
******************************************/
static double timedifference_msec(struct timeval t0, struct timeval t1)
{
    return (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_usec - t0.tv_usec) / 1000.0;
}

int main(int argc, char* argv[])
{
    int ret = 0;
    if(ret = parse_argument(argc, argv)) {
        return ret;
    }

    // setup ONNX runtime env
    prepare_ONNX_Runtime();

    // preprocessing
    preprocess_input(); 

    // run inference
    run_model();

    // postprocessing
    postprocess();

    return 0;
}
