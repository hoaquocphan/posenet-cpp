WORK_DIR := ${CURDIR}
#onnxruntime: PoseNetDemo.cpp
onnxruntime: PoseNetDemo.cpp
	${CXX} -std=c++14 PoseNetDemo.cpp \
	-DONNX_ML \
	-I /media/sf_Ubuntu/posenet/posenet_demo/ \
	-L /data/onnxruntime/build/Linux/RelWithDebInfo/ \
	-L /data/onnxruntime/build/Linux/RelWithDebInfo/onnx/ \
	-L /data/onnxruntime/build/Linux/RelWithDebInfo/external/nsync/ \
	-L /data/onnxruntime/build/Linux/RelWithDebInfo/external/protobuf/cmake/ \
	-L /data/onnxruntime/build/Linux/RelWithDebInfo/external/re2/ \
	-L /data/onnxruntime/cmake/ \
	-L /usr/lib/x86_64-linux-gnu/ \
	-lonnxruntime_session \
	-lonnxruntime_providers \
	-lonnxruntime_framework \
	-lonnxruntime_optimizer \
	-lonnxruntime_graph \
	-lonnxruntime_common \
	-lonnx_proto \
	-lnsync_cpp \
	-lprotobuf \
	-lre2 \
	-lonnxruntime_util \
	-lonnxruntime_mlas \
	-lonnx \
	-ljpeg -ltbb -ltiff -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_photo -lopencv_imgcodecs \
	-lpthread -O2 -fopenmp -ldl ${LDFLAGS} -o PoseNetDemo

clean:
	rm -rf *.o PoseNetDemo 
