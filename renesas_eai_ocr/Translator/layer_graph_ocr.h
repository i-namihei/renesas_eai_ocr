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
* File Name    : layer_graph_ocr.h
* Version      : 1.00
* Description  : Declarations of all functions
***********************************************************************************************************************/
/**********************************************************************************************************************
* History : DD.MM.YYYY Version  Description
*         : 16.06.2017 1.00     First Release
***********************************************************************************************************************/

#ifndef LAYER_GRAPH_OCR_H_
#define LAYER_GRAPH_OCR_H_

void padding(TPrecision *, TPrecision *, TsInt *, TsInt *);
void convolution(TPrecision *,TPrecision *,const TPrecision *,const TPrecision *,TPrecision *,TsInt *,TsInt *);
void max_pooling(TPrecision *, TPrecision *, TPrecision *, TsInt *, TsInt *);
void average_pooling(TPrecision *, TPrecision *, TPrecision *, TsInt *, TsInt *);
void avgpool_padding(TPrecision *, TPrecision *, TsInt *, TsInt *);
void pooling(TPrecision *, TPrecision *, TPrecision *, TsInt *, TsInt *);
void innerproduct(TPrecision *,const TPrecision *,const TPrecision *,TPrecision *,TsInt *,TsInt *);
void batchnormalization4D(TPrecision *,const TPrecision *,const TPrecision *,const TPrecision *,const TPrecision *,TPrecision *,TFloat *,TsInt *);
void relu(TPrecision *, TPrecision *, TsInt, TsInt *);
void relu6(TPrecision *, TPrecision *, TsInt, TsInt *);
void softmax(TPrecision *, TPrecision *, TsInt, TsInt *);

#endif
