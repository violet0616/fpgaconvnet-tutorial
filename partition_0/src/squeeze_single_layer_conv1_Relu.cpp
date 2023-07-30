#include "squeeze_single_layer_conv1_Relu.hpp"

void squeeze_single_layer_conv1_Relu(
    stream_t(squeeze_single_layer_conv1_Relu_data_t) in[SQUEEZE_SINGLE_LAYER_CONV1_RELU_COARSE_IN],
    stream_t(squeeze_single_layer_conv1_Relu_data_t) out[SQUEEZE_SINGLE_LAYER_CONV1_RELU_COARSE_OUT],
    int mode
)
{

#pragma HLS INLINE OFF

#pragma HLS STREAM variable=in depth=2
#pragma HLS STREAM variable=out

#pragma HLS ARRAY_PARTITION variable=in  complete dim=0
#pragma HLS ARRAY_PARTITION variable=out complete dim=0

#pragma HLS DATAFLOW


    squeeze<
        SQUEEZE_SINGLE_LAYER_CONV1_RELU_SQUEEZE_BATCH_SIZE,
        SQUEEZE_SINGLE_LAYER_CONV1_RELU_SQUEEZE_ROWS,
        SQUEEZE_SINGLE_LAYER_CONV1_RELU_SQUEEZE_COLS,
        SQUEEZE_SINGLE_LAYER_CONV1_RELU_SQUEEZE_CHANNELS,
        SQUEEZE_SINGLE_LAYER_CONV1_RELU_SQUEEZE_COARSE_IN,
        SQUEEZE_SINGLE_LAYER_CONV1_RELU_SQUEEZE_COARSE_OUT,
        squeeze_single_layer_conv1_Relu_data_t
    >(in,out);


}

