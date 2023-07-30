#include "single_layer_top.hpp"


static single_layer_conv1_Conv2D_weight_t single_layer_conv1_Conv2D_weights[SINGLE_LAYER_CONV1_CONV2D_COARSE_IN*SINGLE_LAYER_CONV1_CONV2D_COARSE_GROUP][SINGLE_LAYER_CONV1_CONV2D_COARSE_OUT][DIVIDE(SINGLE_LAYER_CONV1_CONV2D_WEIGHTS,SINGLE_LAYER_CONV1_CONV2D_COARSE_IN*SINGLE_LAYER_CONV1_CONV2D_COARSE_GROUP*SINGLE_LAYER_CONV1_CONV2D_COARSE_OUT*SINGLE_LAYER_CONV1_CONV2D_KERNEL_SIZE_X*SINGLE_LAYER_CONV1_CONV2D_KERNEL_SIZE_Y)][SINGLE_LAYER_CONV1_CONV2D_KERNEL_SIZE_X][SINGLE_LAYER_CONV1_CONV2D_KERNEL_SIZE_Y] = {
#include "single_layer_conv1_Conv2D_weights_0.csv"
};
        



#if SINGLE_LAYER_WEIGHTS_RELOADING_FLAG
void reload_weights(
    int weights_reloading_index,
    volatile mem_int wr_hw[SINGLE_LAYER_PORTS_WR][SINGLE_LAYER_SIZE_WR],
    single_layer_conv1_Conv2D_weight_t weights[SINGLE_LAYER_WR_COARSE_IN*SINGLE_LAYER_WR_COARSE_GROUP][SINGLE_LAYER_WR_COARSE_OUT][DIVIDE(SINGLE_LAYER_WR_WEIGHTS,SINGLE_LAYER_WR_COARSE_IN*SINGLE_LAYER_WR_COARSE_GROUP*SINGLE_LAYER_WR_COARSE_OUT*SINGLE_LAYER_WR_KERNEL_SIZE_X*SINGLE_LAYER_WR_KERNEL_SIZE_Y)][SINGLE_LAYER_WR_KERNEL_SIZE_X][SINGLE_LAYER_WR_KERNEL_SIZE_Y]
)
{

#pragma HLS INLINE OFF
#pragma HLS DATAFLOW

#pragma HLS stable variable=weights

    // stream init
    stream_t(single_layer_conv1_Conv2D_weight_t) wr[SINGLE_LAYER_STREAMS_WR];
#pragma HLS STREAM variable=wr
#pragma HLS ARRAY_PARTITION variable=wr complete dim=0

    mem_read<
        SINGLE_LAYER_WR_BATCH_SIZE,
        SINGLE_LAYER_WR_ROWS_IN,
        SINGLE_LAYER_WR_COLS_IN,
        SINGLE_LAYER_WR_CHANNELS_IN,
        SINGLE_LAYER_WR_PORTS_IN,
        SINGLE_LAYER_WR_STREAMS_IN,
        single_layer_conv1_Conv2D_weight_t
    >(wr_hw,wr);

    weights_reloading<
       SINGLE_LAYER_WR_WEIGHTS,
       SINGLE_LAYER_WR_COARSE_IN,
       SINGLE_LAYER_WR_COARSE_OUT,
       SINGLE_LAYER_WR_COARSE_GROUP,
       SINGLE_LAYER_WR_KERNEL_SIZE_X,
       SINGLE_LAYER_WR_KERNEL_SIZE_Y,
       single_layer_conv1_Conv2D_weight_t
    >(wr[0],weights);
}
#endif

void process(
    int weights_reloading_index,
    volatile mem_int in_hw[SINGLE_LAYER_PORTS_IN][SINGLE_LAYER_SIZE_IN],
    volatile mem_int out_hw[SINGLE_LAYER_PORTS_OUT][SINGLE_LAYER_SIZE_OUT]
)
{

#pragma HLS INLINE OFF
#pragma HLS DATAFLOW


#pragma HLS ARRAY_PARTITION variable=single_layer_conv1_Conv2D_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=single_layer_conv1_Conv2D_weights complete dim=2
#pragma HLS RESOURCE variable=single_layer_conv1_Conv2D_weights core=RAM
#pragma HLS STABLE variable=single_layer_conv1_Conv2D_weights
        


    stream_t(single_layer_conv1_Conv2D_input_t) in[SINGLE_LAYER_CONV1_CONV2D_COARSE_IN];
#pragma HLS STREAM variable=in
#pragma HLS ARRAY_PARTITION variable=in complete dim=0
        

    stream_t(single_layer_conv1_Relu_input_t) single_layer_conv1_Conv2D_single_layer_conv1_Relu[SINGLE_LAYER_CONV1_RELU_COARSE_IN];
#pragma HLS STREAM variable=single_layer_conv1_Conv2D_single_layer_conv1_Relu
#pragma HLS ARRAY_PARTITION variable=single_layer_conv1_Conv2D_single_layer_conv1_Relu complete dim=0
        

    stream_t(squeeze_single_layer_conv1_Relu_input_t) single_layer_conv1_Relu_squeeze_single_layer_conv1_Relu[SQUEEZE_SINGLE_LAYER_CONV1_RELU_COARSE_IN];
#pragma HLS STREAM variable=single_layer_conv1_Relu_squeeze_single_layer_conv1_Relu
#pragma HLS ARRAY_PARTITION variable=single_layer_conv1_Relu_squeeze_single_layer_conv1_Relu complete dim=0
        

    stream_t(squeeze_single_layer_conv1_Relu_output_t) out[SQUEEZE_SINGLE_LAYER_CONV1_RELU_COARSE_OUT];
#pragma HLS STREAM variable=out
#pragma HLS ARRAY_PARTITION variable=out complete dim=0
        

    mem_read<
        SINGLE_LAYER_BATCH_SIZE,
        SINGLE_LAYER_ROWS_IN,
        SINGLE_LAYER_COLS_IN,
        SINGLE_LAYER_CHANNELS_IN,
        SINGLE_LAYER_PORTS_IN,
        SINGLE_LAYER_STREAMS_IN,
        single_layer_input_t
    >(in_hw,in);

    int mode = 0;

    single_layer_conv1_Conv2D(single_layer_conv1_Conv2D_weights, in, single_layer_conv1_Conv2D_single_layer_conv1_Relu, mode);
    single_layer_conv1_Relu(single_layer_conv1_Conv2D_single_layer_conv1_Relu, single_layer_conv1_Relu_squeeze_single_layer_conv1_Relu, mode);
    squeeze_single_layer_conv1_Relu(single_layer_conv1_Relu_squeeze_single_layer_conv1_Relu, out, mode);


    mem_write<
        SINGLE_LAYER_BATCH_SIZE,
        SINGLE_LAYER_ROWS_OUT,
        SINGLE_LAYER_COLS_OUT,
        SINGLE_LAYER_CHANNELS_OUT,
        SINGLE_LAYER_PORTS_OUT,
        SINGLE_LAYER_STREAMS_OUT,
        SINGLE_LAYER_WEIGHTS_RELOADING_FACTOR,
        single_layer_output_t
    >(weights_reloading_index,out,out_hw);

}

void fpgaconvnet_ip(
    int mode,
    int weights_reloading_index,
#if SINGLE_LAYER_WEIGHTS_RELOADING_FLAG
    volatile mem_int wr_hw[SINGLE_LAYER_PORTS_WR][SINGLE_LAYER_SIZE_WR],
#endif
    volatile mem_int in_hw[SINGLE_LAYER_PORTS_IN][SINGLE_LAYER_SIZE_IN],
    volatile mem_int out_hw[SINGLE_LAYER_PORTS_OUT][SINGLE_LAYER_SIZE_OUT]
)
{
//#pragma HLS INTERFACE s_axilite port=return                     bundle=ctrl
//#pragma HLS INTERFACE s_axilite port=mode                       bundle=ctrl
//#pragma HLS INTERFACE s_axilite port=weights_reloading_index    bundle=ctrl

#if SINGLE_LAYER_WEIGHTS_RELOADING_FLAG
#pragma HLS ARRAY_PARTITION variable=wr_hw  complete dim=1
#endif
#pragma HLS ARRAY_PARTITION variable=in_hw  complete dim=1
#pragma HLS ARRAY_PARTITION variable=out_hw complete dim=1

#if SINGLE_LAYER_WEIGHTS_RELOADING_FLAG
    const unsigned size_wr  = SINGLE_LAYER_SIZE_WR ;
#endif
    const unsigned size_in  = SINGLE_LAYER_SIZE_IN ;
    const unsigned size_out = SINGLE_LAYER_SIZE_OUT;

#if SINGLE_LAYER_WEIGHTS_RELOADING_FLAG
#pragma HLS INTERFACE m_axi port=wr_hw  offset=slave depth=size_wr  num_read_outstanding=1 num_write_outstanding=1 max_read_burst_length=256 max_write_burst_length=256 name=fpgaconvnet_wr  bundle=fpgaconvnet_port_wr
#endif

#pragma HLS INTERFACE axis register both depth=784 port=in_hw name=fpgaconvnet_in

#pragma HLS INTERFACE axis register both depth=256 port=out_hw name=fpgaconvnet_out


    #pragma HLS DATAFLOW
    if ( mode == 0 ) {
        process(weights_reloading_index,in_hw,out_hw);
    } else if ( mode == 1 ) {
#if SINGLE_LAYER_WEIGHTS_RELOADING_FLAG
        reload_weights(weights_reloading_index,wr_hw,single_layer_conv1_Conv2D_weights);
#endif
    }

}
