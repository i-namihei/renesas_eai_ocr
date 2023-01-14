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
* File Name    : layer_shapes_ocr.h
* Version      : 1.00
* Description  : Initializations
***********************************************************************************************************************/
/**********************************************************************************************************************
* History : DD.MM.YYYY Version  Description
*         : 16.06.2017 1.00     First Release
***********************************************************************************************************************/

#include "Typedef.h"
#include <stdlib.h>
#ifndef LAYER_SHAPES_OCR_H_
#define LAYER_SHAPES_OCR_H_
 
TsOUT* dnn_compute_ocr(TsIN*, TsInt*);
 
TsOUT dnn_buffer1[32448];
TsOUT dnn_buffer2[8112];
 
struct shapes_ocr{
    TsInt Conv22__ocr_shape[16];
    TsInt Relu23__ocr_shape;
    TsInt MaxPool24__ocr_shape[15];
    TFloat BatchNormalization25__ocr_shape[6];
    TsInt Conv26__ocr_shape[16];
    TsInt MaxPool27__ocr_shape[15];
    TsInt Clip28__ocr_shape;
    TsInt Conv29__ocr_shape[16];
    TsInt Relu30__ocr_shape;
    TsInt Conv31__ocr_shape[16];
    TsInt Relu32__ocr_shape;
    TsInt Conv33__ocr_shape[16];
    TsInt Relu34__ocr_shape;
    TsInt MaxPool35__ocr_shape[15];
    TsInt Gemm37__ocr_shape[4];
    TsInt Relu38__ocr_shape;
    TsInt Gemm39__ocr_shape[4];
    TsInt Relu40__ocr_shape;
    TsInt Gemm41__ocr_shape[4];
    TsInt Softmax42__ocr_shape;
};
 
struct shapes_ocr layer_shapes_ocr ={
    {1,3,100,100,12,3,11,11,24,24,2,2,2,2,4,4},
    6912,
    {1,12,24,24,12,12,1,1,1,1,3,3,2,2,0},
    {1,12,12,12,1e-05,0},
    {1,12,12,12,5,12,3,3,12,12,1,1,1,1,1,1},
    {1,5,12,12,6,6,1,1,1,1,3,3,2,2,0},
    180,
    {1,5,6,6,10,5,3,3,6,6,1,1,1,1,1,1},
    360,
    {1,10,6,6,7,10,3,3,6,6,1,1,1,1,1,1},
    252,
    {1,7,6,6,7,7,3,3,6,6,1,1,1,1,1,1},
    252,
    {1,7,6,6,3,3,1,1,1,1,3,3,2,2,0},
    {1,63,63,240},
    240,
    {1,240,240,50},
    50,
    {1,50,50,17},
    17
};
 
#endif
