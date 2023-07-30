#include "single_layer_conv1_Relu.hpp"

void single_layer_conv1_Relu_relu(
    stream_t(single_layer_conv1_Relu_data_t) &in,
    stream_t(single_layer_conv1_Relu_data_t) &out
) {

#pragma HLS INLINE OFF

    relu<
        SINGLE_LAYER_CONV1_RELU_RELU_BATCH_SIZE,
        SINGLE_LAYER_CONV1_RELU_RELU_ROWS,
        SINGLE_LAYER_CONV1_RELU_RELU_COLS,
        SINGLE_LAYER_CONV1_RELU_RELU_CHANNELS,
        single_layer_conv1_Relu_data_t
    >(in,out);

}


void single_layer_conv1_Relu(
    stream_t(single_layer_conv1_Relu_data_t) in[SINGLE_LAYER_CONV1_RELU_COARSE],
    stream_t(single_layer_conv1_Relu_data_t) out[SINGLE_LAYER_CONV1_RELU_COARSE],
    int mode
)
{

#pragma HLS INLINE OFF

#pragma HLS STREAM variable=in depth=2
#pragma HLS STREAM variable=out

#pragma HLS ARRAY_PARTITION variable=in  complete dim=0
#pragma HLS ARRAY_PARTITION variable=out complete dim=0

#pragma HLS DATAFLOW

    for(unsigned int coarse_index=0; coarse_index<SINGLE_LAYER_CONV1_RELU_COARSE; coarse_index++)
    {
#pragma HLS unroll
        single_layer_conv1_Relu_relu(in[coarse_index], out[coarse_index]);
    }
}

