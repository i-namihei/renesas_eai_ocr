/***********************************************************************************************************************
* DISCLAIMER
* This software is supplied by Renesas Electronics Corporation and is only intended for use with Renesas products. No 
* other uses are authorized. This software is owned by Renesas Electronics Corporation and is protected under all 
* applicable laws, including copyright laws. 
* THIS SOFTWARE IS PROVIDED 'AS IS' AND RENESAS MAKES NO WARRANTIES REGARDING
* THIS SOFTWARE, WHETHER EXPRESS, IMPLIED OR STATUTORY, INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, 
* FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. ALL SUCH WARRANTIES ARE EXPRESSLY DISCLAIMED. TO THE MAXIMUM
* EXTENT PERMITTED NOT PROHIBITED BY LAW, NEITHER RENESAS ELECTRONICS CORPORATION NOR ANY OF ITS AFFILIATED COMPANIES 
* SHALL BE LIABLE FOR ANY DIRECT, INDIRECT, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES FOR ANY REASON RELATED TO THIS 
* SOFTWARE, EVEN IF RENESAS OR ITS AFFILIATES HAVE BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
* Renesas reserves the right, without notice, to make changes to this software and to discontinue the availability of 
* this software. By using this software, you agree to the additional terms and conditions found by accessing the 
* following link:
* http://www.renesas.com/disclaimer 
*
* Changed from original python code to C source code.
* Copyright (C) 2017 Renesas Electronics Corporation. All rights reserved.
***********************************************************************************************************************/
/***********************************************************************************************************************
* File Name    : dnn_compute_ocr.c
* Version      : 1.00
* Description  : The function calls
***********************************************************************************************************************/
/**********************************************************************************************************************
* History : DD.MM.YYYY Version  Description
*         : 16.06.2017 1.00     First Release
***********************************************************************************************************************/

 
#include "layer_shapes_ocr.h"
#include "layer_graph_ocr.h"
#include "weights_ocr.h"
 
TsOUT* dnn_compute_ocr(TsIN* input_layer, TsInt *errorcode)
{
  *errorcode = 0;
  convolution(input_layer,dnn_buffer1,Conv22__ocr_weights,Conv22__ocr_biases,dnn_buffer2,layer_shapes_ocr.Conv22__ocr_shape,errorcode);
  relu(dnn_buffer2,dnn_buffer2,layer_shapes_ocr.Relu23__ocr_shape,errorcode);
  pooling(dnn_buffer2,dnn_buffer1,dnn_buffer2,layer_shapes_ocr.MaxPool24__ocr_shape,errorcode);
  batchnormalization4D(dnn_buffer2,BatchNormalization25__ocr_moving_mean,BatchNormalization25__ocr_moving_variance,BatchNormalization25__ocr_beta,BatchNormalization25__ocr_gamma,dnn_buffer2,layer_shapes_ocr.BatchNormalization25__ocr_shape,errorcode);
  convolution(dnn_buffer2,dnn_buffer1,Conv26__ocr_weights,Conv26__ocr_biases,dnn_buffer2,layer_shapes_ocr.Conv26__ocr_shape,errorcode);
  pooling(dnn_buffer2,dnn_buffer1,dnn_buffer2,layer_shapes_ocr.MaxPool27__ocr_shape,errorcode);
  relu6(dnn_buffer2,dnn_buffer2,layer_shapes_ocr.Clip28__ocr_shape,errorcode);
  convolution(dnn_buffer2,dnn_buffer1,Conv29__ocr_weights,Conv29__ocr_biases,dnn_buffer2,layer_shapes_ocr.Conv29__ocr_shape,errorcode);
  relu(dnn_buffer2,dnn_buffer2,layer_shapes_ocr.Relu30__ocr_shape,errorcode);
  convolution(dnn_buffer2,dnn_buffer1,Conv31__ocr_weights,Conv31__ocr_biases,dnn_buffer2,layer_shapes_ocr.Conv31__ocr_shape,errorcode);
  relu(dnn_buffer2,dnn_buffer2,layer_shapes_ocr.Relu32__ocr_shape,errorcode);
  convolution(dnn_buffer2,dnn_buffer1,Conv33__ocr_weights,Conv33__ocr_biases,dnn_buffer2,layer_shapes_ocr.Conv33__ocr_shape,errorcode);
  relu(dnn_buffer2,dnn_buffer2,layer_shapes_ocr.Relu34__ocr_shape,errorcode);
  pooling(dnn_buffer2,dnn_buffer1,dnn_buffer2,layer_shapes_ocr.MaxPool35__ocr_shape,errorcode);
   
  innerproduct(dnn_buffer2,Gemm37__ocr_weights,Gemm37__ocr_biases,dnn_buffer1,layer_shapes_ocr.Gemm37__ocr_shape,errorcode);
  relu(dnn_buffer1,dnn_buffer1,layer_shapes_ocr.Relu38__ocr_shape,errorcode);
   
  innerproduct(dnn_buffer1,Gemm39__ocr_weights,Gemm39__ocr_biases,dnn_buffer2,layer_shapes_ocr.Gemm39__ocr_shape,errorcode);
  relu(dnn_buffer2,dnn_buffer2,layer_shapes_ocr.Relu40__ocr_shape,errorcode);
   
  innerproduct(dnn_buffer2,Gemm41__ocr_weights,Gemm41__ocr_biases,dnn_buffer1,layer_shapes_ocr.Gemm41__ocr_shape,errorcode);
  softmax(dnn_buffer1,dnn_buffer1,layer_shapes_ocr.Softmax42__ocr_shape,errorcode);
  return(dnn_buffer1);
}
