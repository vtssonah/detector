cmake_minimum_required(VERSION 2.8)

ADD_DEFINITIONS(
    -std=c++11 # Or -std=c++0x
    # Other flags
)
project( main )
find_package( OpenCV )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( main main.cpp HarrisPC.cpp image_alg.cpp NaiveGaussianHOGPC.cpp ParkingClassifier.cpp ParkingDatasetDescription.cpp Parkplatz.cpp SVMParkingClassifier.cpp util.cpp)
target_link_libraries( main ${OpenCV_LIBS} -I/usr/local/include/opencv -I/usr/local/include  -L/usr/local/lib -lopencv_shape -lopencv_stitching -lopencv_objdetect -lopencv_superres -lopencv_videostab -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_video -lopencv_photo -lopencv_ml -lopencv_imgproc -lopencv_flann -lopencv_core  -lboost_system -lboost_filesystem -lboost_iostreams -lboost_system -lboost_filesystem)
include_directories(${.})
