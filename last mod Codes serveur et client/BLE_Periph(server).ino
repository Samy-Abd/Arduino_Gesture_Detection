/*
  BLE_Peripheral.ino

  This program uses the ArduinoBLE library to set-up an Arduino Nano 33 BLE 
  as a peripheral device and specifies a service and a characteristic. Depending 
  of the value of the specified characteristic, an on-board LED gets on. 

  The circuit:
  - Arduino Nano 33 BLE. 

  This example code is in the public domain.
*/

#include <ArduinoBLE.h>
#include <Arduino_LSM9DS1.h>

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

#include "model left.h"

const float accelerationThreshold = 2.5;  // threshold of significant in G's
const int numSamples = 119;

int samplesRead = numSamples;

// global variables used for TensorFlow Lite (Micro)
tflite::MicroErrorReporter tflErrorReporter;

// pull in all the TFLM ops, you can remove this line and
// only pull in the TFLM ops you need, if would like to reduce
// the compiled size of the sketch.
tflite::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Create a static memory buffer for TFLM, the size may need to
// be adjusted based on the model you are using
constexpr int tensorArenaSize = 8 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

// array to map gesture index to a name
const char* GESTURES[] = {
  "clap",
  "wakanda"
};

#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))


enum {
  GESTURE_UP = 0,
  GESTURE_DOWN = 1,
};

const char* deviceServiceUuid = "19b10000-e8f2-537e-4f6c-d104768a1214";
const char* deviceServiceCharacteristicUuid = "19b10001-e8f2-537e-4f6c-d104768a1214";

int clientGesture = -1;

BLEService gestureService(deviceServiceUuid);
BLEByteCharacteristic gestureCharacteristic(deviceServiceCharacteristicUuid, BLERead | BLEWrite);


void setup() {
  Serial.begin(9600);
  while (!Serial)
    ;

  // initialize the IMU
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1)
      ;
  }

  // print out the samples rates of the IMUs
  Serial.print("Accelerometer sample rate = ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println(" Hz");
  Serial.print("Gyroscope sample rate = ");
  Serial.print(IMU.gyroscopeSampleRate());
  Serial.println(" Hz");

  Serial.println();

  // get the TFL representation of the model byte array
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1)
      ;
  }

  // Create an interpreter to run the model
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);

  // Allocate memory for the model's input and output tensors
  tflInterpreter->AllocateTensors();

  // Get pointers for the model's input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);

  //INIT LED
  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);
  pinMode(LED_BUILTIN, OUTPUT);

  digitalWrite(LEDR, HIGH);
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDB, HIGH);
  digitalWrite(LED_BUILTIN, LOW);


  if (!BLE.begin()) {
    //Serial.println("- Starting Bluetooth® Low Energy module failed!");
    while (1)
      ;
  }

  BLE.setLocalName("Arduino Nano 33 BLE (Peripheral)");
  BLE.setAdvertisedService(gestureService);
  gestureService.addCharacteristic(gestureCharacteristic);
  BLE.addService(gestureService);
  gestureCharacteristic.writeValue(-1);
  BLE.advertise();

  //Serial.println("Nano 33 BLE (Peripheral Device)");
  //Serial.println(" ");
}

void loop() {
   Serial.println("Starting model...");
      float aX, aY, aZ, gX, gY, gZ;
      // wait for significant motion
      while (samplesRead == numSamples) {
        if (IMU.accelerationAvailable()) {
          // read the acceleration data
          IMU.readAcceleration(aX, aY, aZ);

          // sum up the absolutes
          float aSum = fabs(aX) + fabs(aY) + fabs(aZ);

          // check if it's above the threshold
          if (aSum >= accelerationThreshold) {
            // reset the sample read count
            samplesRead = 0;
            break;
          }
        }
      }
      //Serial.println("Detected significant motion...");
      while (samplesRead < numSamples) {
       // Serial.println("samplesRead < numSamples...");
        // check if new acceleration AND gyroscope data is available
        if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
          // read the acceleration and gyroscope data
          IMU.readAcceleration(aX, aY, aZ);
          IMU.readGyroscope(gX, gY, gZ);
        //  Serial.println("Read gyroscopes...");
          // normalize the IMU data between 0 to 1 and store in the model's
          // input tensor
          tflInputTensor->data.f[samplesRead * 6 + 0] = (aX + 4.0) / 8.0;
          tflInputTensor->data.f[samplesRead * 6 + 1] = (aY + 4.0) / 8.0;
          tflInputTensor->data.f[samplesRead * 6 + 2] = (aZ + 4.0) / 8.0;
          tflInputTensor->data.f[samplesRead * 6 + 3] = (gX + 2000.0) / 4000.0;
          tflInputTensor->data.f[samplesRead * 6 + 4] = (gY + 2000.0) / 4000.0;
          tflInputTensor->data.f[samplesRead * 6 + 5] = (gZ + 2000.0) / 4000.0;

          samplesRead++;
         // Serial.println("Added Tensor...");
          if (samplesRead == numSamples) {
           // Serial.println("Finished reading all samples...");
            // Run inferencing
            Serial.println("Invoking model...");
            TfLiteStatus invokeStatus = tflInterpreter->Invoke();
            Serial.println("Finished invoking model...");
            if (invokeStatus != kTfLiteOk) {
              Serial.println("Invoke failed!");
              while (1)
                ;
            }
            // Loop through the output tensor values from the model
            for (int i = 0; i < NUM_GESTURES; i++) {
              Serial.print(GESTURES[i]);
              Serial.print(": ");
              Serial.println(tflOutputTensor->data.f[i], 6);
            }
            Serial.println();
          }
        }
      }
     // Serial.println("After Server model deteecton...");
      int maxIndex = 0;
      for (int i = 1; i < NUM_GESTURES; ++i) {
        if (tflOutputTensor->data.f[maxIndex] < tflOutputTensor->data.f[i]) {
          maxIndex = i;
        }
      }
     // Serial.println("After argmax...");
      switch (maxIndex) {
        case 0:
          Serial.println("-Server : Clap gesture detected");
          break;
        case 1:
          Serial.println("-Server : Wakanda gesture detected");
          break;
      }

           



  BLEDevice central = BLE.central();
  //Serial.println("- Discovering central device...");

  while(!central) {
    //Serial.println("* Connected to central device!");
   // Serial.print("* Device MAC address: ");
   Serial.println("waiting for client...");
    central = BLE.central();
  // Serial.println("* Disconnected to central device!");
  }
      if(central)
      {
      Serial.println(central.address());
          Serial.println(" ");

          while (central.connected()) {
            if (gestureCharacteristic.written()) {
              
              clientGesture = gestureCharacteristic.value();
              writeGesture(clientGesture, maxIndex);
            }
          }
      }   

}

void writeGesture(int clientGesture, int serverGesture) {

  if (clientGesture != serverGesture) {
   // Serial.println("* Unrecognized gesture!");
    //Serial.println(" ");
    digitalWrite(LEDR, LOW);
    digitalWrite(LEDG, HIGH);
    digitalWrite(LEDB, HIGH);
    digitalWrite(LED_BUILTIN, LOW);
  } else {
    switch (clientGesture) {
      case 0:
       // Serial.println("* Server and Client : Clap gesture detected");
       // Serial.println(" ");
        digitalWrite(LEDR, HIGH);
        digitalWrite(LEDG, HIGH);
        digitalWrite(LEDB, LOW);
        digitalWrite(LED_BUILTIN, LOW);
        break;
      case 1:
       // Serial.println("* Server and Client : Wakanda gesture detected");
       // Serial.println(" ");
        digitalWrite(LEDR, HIGH);
        digitalWrite(LEDG, LOW);
        digitalWrite(LEDB, HIGH);
        digitalWrite(LED_BUILTIN, LOW);
        break;
      default:
        digitalWrite(LEDR, HIGH);
        digitalWrite(LEDG, HIGH);
        digitalWrite(LEDB, HIGH);
        digitalWrite(LED_BUILTIN, LOW);
        break;
    }
  }
}