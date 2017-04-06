################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/HarrisPC.cpp \
../src/NaiveGaussianHOGPC.cpp \
../src/ParkingClassifier.cpp \
../src/ParkingDatasetDescription.cpp \
../src/Parkplatz.cpp \
../src/SVMParkingClassifier.cpp \
../src/image_alg.cpp \
../src/main.cpp \
../src/util.cpp 

OBJS += \
./src/HarrisPC.o \
./src/NaiveGaussianHOGPC.o \
./src/ParkingClassifier.o \
./src/ParkingDatasetDescription.o \
./src/Parkplatz.o \
./src/SVMParkingClassifier.o \
./src/image_alg.o \
./src/main.o \
./src/util.o 

CPP_DEPS += \
./src/HarrisPC.d \
./src/NaiveGaussianHOGPC.d \
./src/ParkingClassifier.d \
./src/ParkingDatasetDescription.d \
./src/Parkplatz.d \
./src/SVMParkingClassifier.d \
./src/image_alg.d \
./src/main.d \
./src/util.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I"/home/timo/workspaces/Sonah/Detectors/include" -I/usr/include/opencv -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


