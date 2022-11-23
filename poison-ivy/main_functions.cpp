/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "main_functions.h"
#include <LCD_st7735.h>
#include <hardware/gpio.h>
#include <hardware/irq.h>
#include <hardware/uart.h>
#include <pico/stdio_usb.h>

#include "detection_responder.h"
#include "image_provider.h"
#include "edge-impulse-sdk/tensorflow/lite/micro/micro_error_reporter.h"
#include "edge-impulse-sdk/classifier/ei_run_classifier.h"

const uint LED_PIN = 25;
#define UART_ID uart0
#define BAUD_RATE 115200
#define DATA_BITS 8
#define STOP_BITS 1
#define PARITY UART_PARITY_NONE
#define UART_TX_PIN 0
#define UART_RX_PIN 1

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter *   error_reporter = nullptr;
int8_t image[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE] = {0};
float image_output[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE] = {0};
ei_impulse_result_t result = { 0 };

static bool debug_nn = false;

}  // namespace

#ifndef DO_NOT_OUTPUT_TO_UART
// RX interrupt handler
void on_uart_rx() {
  char cameraCommand = 0;
  while (uart_is_readable(UART_ID)) {
    cameraCommand = uart_getc(UART_ID);
    // Can we send it back?
    if (uart_is_writable(UART_ID)) {
      uart_putc(UART_ID, cameraCommand);
    }
  }
}

void setup_uart() {
  // Set up our UART with the required speed.
  uint baud = uart_init(UART_ID, BAUD_RATE);
  // Set the TX and RX pins by using the function select on the GPIO
  // Set datasheet for more information on function select
  gpio_set_function(UART_TX_PIN, GPIO_FUNC_UART);
  gpio_set_function(UART_RX_PIN, GPIO_FUNC_UART);
  // Set our data format
  uart_set_format(UART_ID, DATA_BITS, STOP_BITS, PARITY);
  // Turn off FIFO's - we want to do this character by character
  uart_set_fifo_enabled(UART_ID, false);
  // Set up a RX interrupt
  // We need to set up the handler first
  // Select correct interrupt for the UART we are using
  int UART_IRQ = UART_ID == uart0 ? UART0_IRQ : UART1_IRQ;

  // And set up and enable the interrupt handlers
  irq_set_exclusive_handler(UART_IRQ, on_uart_rx);
  irq_set_enabled(UART_IRQ, true);

  // Now enable the UART to send interrupts - RX only
  uart_set_irq_enables(UART_ID, true, false);
}
#else
void setup_uart() {}
#endif

//code from EI-generated Arduino library, recommended by Edge Impulse team
int raw_feature_get_data(size_t offset, size_t length, float *out_ptr) {
       
    //size_t pixel_ix = offset * 2;
    size_t pixel_ix = offset; 
    size_t bytes_left = length;
    size_t out_ptr_ix = 0;

    // read byte for byte
    while (bytes_left != 0) {
        // grab the pixel value and convert to r/g/b
        uint16_t pixel   = ST7735_COLOR565(image[pixel_ix], image[pixel_ix], image[pixel_ix]);
         
        uint8_t r, g, b;
        r = ((pixel >> 11) & 0x1f) << 3;
        g = ((pixel >> 5) & 0x3f) << 2;
        b = (pixel & 0x1f) << 3;

        // then convert to out_ptr format
        float pixel_f = (r << 16) + (g << 8) + b;
        out_ptr[out_ptr_ix] = pixel_f;

        // and go to the next pixel
        out_ptr_ix++;
        pixel_ix++;
        bytes_left--;
    }

    // and done!
    return 0;
}

// The name of this function is important for Arduino compatibility.
void setup() {
  
  

  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  
  setup_uart();
  stdio_usb_init();
  
  TfLiteStatus setup_status = ScreenInit(error_reporter);
  if (setup_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Screen Set up failed\n");
    ei_printf("Failed to init screen\n");
  }

}



// The name of this function is important for Arduino compatibility.
void loop() {
  
  // Get image from camera.
  int kNumCols = 96;
  int kNumRows = 96;
  int kNumChannels = 1;
  
  if (kTfLiteOk
      != GetImage(error_reporter, kNumCols, kNumRows, kNumChannels, image)) {
    TF_LITE_REPORT_ERROR(error_reporter, "Image capture failed.");
    ei_printf("Failed to get image\n");
  }

  // Run the model on this input and make sure it succeeds.
  // put Edge Impulse model code here 
  // Turn the raw buffer in a signal which we can the classify
  // this is where we'll write all the output
  signal_t signal;
  signal.total_length = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
  signal.get_data = &raw_feature_get_data;

  // read through the signal buffered, like the classifier lib also does
  for (size_t ix = 0; ix < signal.total_length; ix += 1024) {
     size_t bytes_to_read = 1024;
     if (ix + bytes_to_read > signal.total_length) {
         bytes_to_read = signal.total_length - ix;
     }
     int r = signal.get_data(ix, bytes_to_read, image_output + ix);
  }

 
  // Run the classifier
  int err = run_classifier(&signal, &result, debug_nn);
  if (err != EI_IMPULSE_OK) {
     ei_printf("ERR: Failed to run classifier (%d)\n", err);
     return;
  }
 
  // print the predictions
  ei_printf("Predictions ");
  for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
      ei_printf("    %s: %.5f\n", result.classification[ix].label, result.classification[ix].value);
  }
  // Process the inference results.
  int8_t no_poison_ivy_score    = floor(result.classification[0].value * 100);
  int8_t poison_ivy_score       = floor(result.classification[1].value * 100);
  RespondToDetection(error_reporter, poison_ivy_score, no_poison_ivy_score);
  
  
#if SCREEN
  char array[10];
  sprintf(array, "%d%%", poison_ivy_score);
  ST7735_FillRectangle(10, 120, ST7735_WIDTH, 60, ST7735_BLACK);
  //if the poison ivy score is > 60%, change text color to red as a warning
  //otherwise, text is green
  if(poison_ivy_score > 60)
  {
    ST7735_WriteString(10, 120, array, Font_16x26, ST7735_RED, ST7735_BLACK);
  }
  else
  {
    ST7735_WriteString(10, 120, array, Font_16x26, ST7735_GREEN, ST7735_BLACK);
  }
#endif
  TF_LITE_REPORT_ERROR(error_reporter, "**********");
  ei_sleep(2000);
}
