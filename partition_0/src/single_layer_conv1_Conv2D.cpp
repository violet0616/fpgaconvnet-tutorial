#include "single_layer_conv1_Conv2D.hpp"

void single_layer_conv1_Conv2D_sliding_window(
    stream_t(single_layer_conv1_Conv2D_input_t)  &in,
    stream_t(single_layer_conv1_Conv2D_output_t) out[SINGLE_LAYER_CONV1_CONV2D_KERNEL_SIZE_X][SINGLE_LAYER_CONV1_CONV2D_KERNEL_SIZE_Y]
) {

#pragma HLS INLINE OFF

    sliding_window<
        SINGLE_LAYER_CONV1_CONV2D_SLIDING_WINDOW_BATCH_SIZE,
        SINGLE_LAYER_CONV1_CONV2D_SLIDING_WINDOW_ROWS,
        SINGLE_LAYER_CONV1_CONV2D_SLIDING_WINDOW_COLS,
        SINGLE_LAYER_CONV1_CONV2D_SLIDING_WINDOW_CHANNELS,
        SINGLE_LAYER_CONV1_CONV2D_SLIDING_WINDOW_PAD_TOP,
        SINGLE_LAYER_CONV1_CONV2D_SLIDING_WINDOW_PAD_RIGHT,
        SINGLE_LAYER_CONV1_CONV2D_SLIDING_WINDOW_PAD_BOTTOM,
        SINGLE_LAYER_CONV1_CONV2D_SLIDING_WINDOW_PAD_LEFT,
        SINGLE_LAYER_CONV1_CONV2D_SLIDING_WINDOW_STRIDE_X,
        SINGLE_LAYER_CONV1_CONV2D_SLIDING_WINDOW_STRIDE_Y,
        SINGLE_LAYER_CONV1_CONV2D_SLIDING_WINDOW_KERNEL_SIZE_X,
        SINGLE_LAYER_CONV1_CONV2D_SLIDING_WINDOW_KERNEL_SIZE_Y,
        single_layer_conv1_Conv2D_input_t
    >(in,out);

}

void single_layer_conv1_Conv2D_fork(
#if SINGLE_LAYER_CONV1_CONV2D_KERNEL_SIZE_X == 1 && SINGLE_LAYER_CONV1_CONV2D_KERNEL_SIZE_Y == 1
    stream_t(single_layer_conv1_Conv2D_input_t)  &in,
    stream_t(single_layer_conv1_Conv2D_output_t) out[SINGLE_LAYER_CONV1_CONV2D_COARSE_OUT]
#else
    stream_t(single_layer_conv1_Conv2D_input_t)  in[SINGLE_LAYER_CONV1_CONV2D_KERNEL_SIZE_X][SINGLE_LAYER_CONV1_CONV2D_KERNEL_SIZE_Y],
    stream_t(single_layer_conv1_Conv2D_output_t) out[SINGLE_LAYER_CONV1_CONV2D_COARSE_OUT][SINGLE_LAYER_CONV1_CONV2D_KERNEL_SIZE_X][SINGLE_LAYER_CONV1_CONV2D_KERNEL_SIZE_Y]
#endif
) {

#pragma HLS INLINE OFF

    fork<
        SINGLE_LAYER_CONV1_CONV2D_FORK_BATCH_SIZE,
        SINGLE_LAYER_CONV1_CONV2D_FORK_ROWS,
        SINGLE_LAYER_CONV1_CONV2D_FORK_COLS,
        SINGLE_LAYER_CONV1_CONV2D_FORK_CHANNELS,
        SINGLE_LAYER_CONV1_CONV2D_FORK_COARSE,
#if SINGLE_LAYER_CONV1_CONV2D_FORK_KERNEL_SIZE_X > 1 || SINGLE_LAYER_CONV1_CONV2D_FORK_KERNEL_SIZE_Y > 1
        SINGLE_LAYER_CONV1_CONV2D_FORK_KERNEL_SIZE_X,
        SINGLE_LAYER_CONV1_CONV2D_FORK_KERNEL_SIZE_Y,
#endif
        single_layer_conv1_Conv2D_input_t
    >(in,out);

}

void single_layer_conv1_Conv2D_conv(
    const single_layer_conv1_Conv2D_weight_t weights[DIVIDE(SINGLE_LAYER_CONV1_CONV2D_WEIGHTS,SINGLE_LAYER_CONV1_CONV2D_COARSE_IN*SINGLE_LAYER_CONV1_CONV2D_COARSE_GROUP*SINGLE_LAYER_CONV1_CONV2D_COARSE_OUT*SINGLE_LAYER_CONV1_CONV2D_KERNEL_SIZE_X*SINGLE_LAYER_CONV1_CONV2D_KERNEL_SIZE_Y)][SINGLE_LAYER_CONV1_CONV2D_KERNEL_SIZE_X][SINGLE_LAYER_CONV1_CONV2D_KERNEL_SIZE_Y],
#if SINGLE_LAYER_CONV1_CONV2D_KERNEL_SIZE_X == 1 && SINGLE_LAYER_CONV1_CONV2D_KERNEL_SIZE_Y == 1
    stream_t(single_layer_conv1_Conv2D_input_t) &in,
#else
    stream_t(single_layer_conv1_Conv2D_input_t)  in[SINGLE_LAYER_CONV1_CONV2D_KERNEL_SIZE_X][SINGLE_LAYER_CONV1_CONV2D_KERNEL_SIZE_Y],
#endif
    stream_t(single_layer_conv1_Conv2D_acc_t) &out
) {

#pragma HLS INLINE OFF

    conv<
        SINGLE_LAYER_CONV1_CONV2D_CONV_BATCH_SIZE,
        SINGLE_LAYER_CONV1_CONV2D_CONV_ROWS,
        SINGLE_LAYER_CONV1_CONV2D_CONV_COLS,
        SINGLE_LAYER_CONV1_CONV2D_CONV_CHANNELS,
        SINGLE_LAYER_CONV1_CONV2D_CONV_FILTERS,
        SINGLE_LAYER_CONV1_CONV2D_CONV_GROUPS,
#if (SINGLE_LAYER_CONV1_CONV2D_CONV_KERNEL_SIZE_X > 1) || (SINGLE_LAYER_CONV1_CONV2D_CONV_KERNEL_SIZE_Y > 1)
        SINGLE_LAYER_CONV1_CONV2D_CONV_FINE,
        SINGLE_LAYER_CONV1_CONV2D_CONV_KERNEL_SIZE_X,
        SINGLE_LAYER_CONV1_CONV2D_CONV_KERNEL_SIZE_Y,
#endif
        single_layer_conv1_Conv2D_input_t,
        single_layer_conv1_Conv2D_weight_t,
        single_layer_conv1_Conv2D_acc_t
    >(in,weights,out);

}

void single_layer_conv1_Conv2D_accum(
    stream_t(single_layer_conv1_Conv2D_acc_t) &in,
    stream_t(single_layer_conv1_Conv2D_acc_t) &out
) {

#pragma HLS INLINE OFF

    accum<
        SINGLE_LAYER_CONV1_CONV2D_ACCUM_BATCH_SIZE,
        SINGLE_LAYER_CONV1_CONV2D_ACCUM_ROWS,
        SINGLE_LAYER_CONV1_CONV2D_ACCUM_COLS,
        SINGLE_LAYER_CONV1_CONV2D_ACCUM_CHANNELS,
        SINGLE_LAYER_CONV1_CONV2D_ACCUM_FILTERS,
        SINGLE_LAYER_CONV1_CONV2D_ACCUM_GROUPS,
        single_layer_conv1_Conv2D_acc_t
    >(in,out);

}

void single_layer_conv1_Conv2D_glue(
    stream_t(single_layer_conv1_Conv2D_acc_t) in[SINGLE_LAYER_CONV1_CONV2D_COARSE_IN*SINGLE_LAYER_CONV1_CONV2D_COARSE_GROUP][SINGLE_LAYER_CONV1_CONV2D_COARSE_OUT],
    stream_t(single_layer_conv1_Conv2D_output_t) out[SINGLE_LAYER_CONV1_CONV2D_COARSE_OUT]
) {

#pragma HLS INLINE OFF

    glue<
        SINGLE_LAYER_CONV1_CONV2D_GLUE_BATCH_SIZE,
        SINGLE_LAYER_CONV1_CONV2D_GLUE_ROWS,
        SINGLE_LAYER_CONV1_CONV2D_GLUE_COLS,
        SINGLE_LAYER_CONV1_CONV2D_GLUE_FILTERS,
        SINGLE_LAYER_CONV1_CONV2D_GLUE_COARSE_IN,
        SINGLE_LAYER_CONV1_CONV2D_GLUE_COARSE_OUT,
        SINGLE_LAYER_CONV1_CONV2D_GLUE_COARSE_GROUP,
        single_layer_conv1_Conv2D_acc_t,
        single_layer_conv1_Conv2D_output_t
    >(in,out);

}

void single_layer_conv1_Conv2D_bias(
    const single_layer_conv1_Conv2D_biases_t biases[SINGLE_LAYER_CONV1_CONV2D_BIAS_FILTERS],
    stream_t(single_layer_conv1_Conv2D_output_t) &in,
    stream_t(single_layer_conv1_Conv2D_output_t) &out
) {

#pragma HLS INLINE OFF

    bias<
        SINGLE_LAYER_CONV1_CONV2D_BIAS_BATCH_SIZE,
        SINGLE_LAYER_CONV1_CONV2D_BIAS_ROWS,
        SINGLE_LAYER_CONV1_CONV2D_BIAS_COLS,
        SINGLE_LAYER_CONV1_CONV2D_BIAS_FILTERS,
        single_layer_conv1_Conv2D_output_t,
        single_layer_conv1_Conv2D_biases_t
    >(in,biases,out);

}

void single_layer_conv1_Conv2D(
    const single_layer_conv1_Conv2D_weight_t weights[SINGLE_LAYER_CONV1_CONV2D_COARSE_IN*SINGLE_LAYER_CONV1_CONV2D_COARSE_GROUP][SINGLE_LAYER_CONV1_CONV2D_COARSE_OUT][DIVIDE(SINGLE_LAYER_CONV1_CONV2D_WEIGHTS,SINGLE_LAYER_CONV1_CONV2D_COARSE_IN*SINGLE_LAYER_CONV1_CONV2D_COARSE_GROUP*SINGLE_LAYER_CONV1_CONV2D_COARSE_OUT*SINGLE_LAYER_CONV1_CONV2D_KERNEL_SIZE_X*SINGLE_LAYER_CONV1_CONV2D_KERNEL_SIZE_Y)][SINGLE_LAYER_CONV1_CONV2D_KERNEL_SIZE_X][SINGLE_LAYER_CONV1_CONV2D_KERNEL_SIZE_Y],
#if SINGLE_LAYER_CONV1_CONV2D_HAS_BIAS == 1
    const single_layer_conv1_Conv2D_biases_t biases[SINGLE_LAYER_CONV1_CONV2D_COARSE_OUT][SINGLE_LAYER_CONV1_CONV2D_BIAS_FILTERS],
#endif
    stream_t(single_layer_conv1_Conv2D_input_t)  in[SINGLE_LAYER_CONV1_CONV2D_COARSE_IN*SINGLE_LAYER_CONV1_CONV2D_COARSE_GROUP],
    stream_t(single_layer_conv1_Conv2D_output_t) out[SINGLE_LAYER_CONV1_CONV2D_COARSE_OUT*SINGLE_LAYER_CONV1_CONV2D_COARSE_GROUP],
    int mode
)
{

#pragma HLS INLINE OFF

#pragma HLS STREAM variable=in depth=2
#pragma HLS STREAM variable=out

#pragma HLS ARRAY_PARTITION variable=in  complete dim=0
#pragma HLS ARRAY_PARTITION variable=out complete dim=0

#pragma HLS DATAFLOW
#pragma HLS stable variable=weights

#if SINGLE_LAYER_CONV1_CONV2D_KERNEL_SIZE_X >= 1 || SINGLE_LAYER_CONV1_CONV2D_KERNEL_SIZE_Y >= 1
    stream_t(single_layer_conv1_Conv2D_input_t) sw_out[SINGLE_LAYER_CONV1_CONV2D_COARSE_IN*SINGLE_LAYER_CONV1_CONV2D_COARSE_GROUP][SINGLE_LAYER_CONV1_CONV2D_KERNEL_SIZE_X][SINGLE_LAYER_CONV1_CONV2D_KERNEL_SIZE_Y];
    #pragma HLS STREAM variable=sw_out
    #pragma HLS ARRAY_PARTITION variable=sw_out complete dim=0
#endif

#if SINGLE_LAYER_CONV1_CONV2D_KERNEL_SIZE_X == 1 && SINGLE_LAYER_CONV1_CONV2D_KERNEL_SIZE_Y == 1
    stream_t(single_layer_conv1_Conv2D_input_t) fork_out[SINGLE_LAYER_CONV1_CONV2D_COARSE_IN*SINGLE_LAYER_CONV1_CONV2D_COARSE_GROUP][SINGLE_LAYER_CONV1_CONV2D_COARSE_OUT];
#else
    stream_t(single_layer_conv1_Conv2D_input_t) fork_out[SINGLE_LAYER_CONV1_CONV2D_COARSE_IN*SINGLE_LAYER_CONV1_CONV2D_COARSE_GROUP][SINGLE_LAYER_CONV1_CONV2D_COARSE_OUT][SINGLE_LAYER_CONV1_CONV2D_KERNEL_SIZE_X][SINGLE_LAYER_CONV1_CONV2D_KERNEL_SIZE_Y];
#endif
    #pragma HLS STREAM variable=fork_out
    #pragma HLS ARRAY_PARTITION variable=fork_out complete dim=0

    stream_t(single_layer_conv1_Conv2D_acc_t) conv_out[SINGLE_LAYER_CONV1_CONV2D_COARSE_IN*SINGLE_LAYER_CONV1_CONV2D_COARSE_GROUP][SINGLE_LAYER_CONV1_CONV2D_COARSE_OUT];
    #pragma HLS STREAM variable=conv_out
    #pragma HLS ARRAY_PARTITION variable=conv_out complete dim=0

#if SINGLE_LAYER_CONV1_CONV2D_ACCUM_CHANNELS > 1
    stream_t(single_layer_conv1_Conv2D_acc_t) accum_out[SINGLE_LAYER_CONV1_CONV2D_COARSE_IN*SINGLE_LAYER_CONV1_CONV2D_COARSE_GROUP][SINGLE_LAYER_CONV1_CONV2D_COARSE_OUT];
    #pragma HLS STREAM variable=accum_out
    #pragma HLS ARRAY_PARTITION variable=accum_out complete dim=0
#endif

#if SINGLE_LAYER_CONV1_CONV2D_HAS_BIAS == 1
    stream_t(single_layer_conv1_Conv2D_output_t) glue_out[SINGLE_LAYER_CONV1_CONV2D_COARSE_OUT];
    #pragma HLS STREAM variable=glue_out
    #pragma HLS ARRAY_PARTITION variable=glue_out complete dim=0
#endif

    single_layer_conv1_Conv2D_coarse_in_loop: for(unsigned int i=0;i<SINGLE_LAYER_CONV1_CONV2D_COARSE_IN*SINGLE_LAYER_CONV1_CONV2D_COARSE_GROUP;i++) {
        #pragma HLS unroll
#if SINGLE_LAYER_CONV1_CONV2D_KERNEL_SIZE_X == 1 && SINGLE_LAYER_CONV1_CONV2D_KERNEL_SIZE_Y == 1
        single_layer_conv1_Conv2D_fork(in[i], fork_out[i]);
#else
        single_layer_conv1_Conv2D_sliding_window(in[i], sw_out[i]);
        single_layer_conv1_Conv2D_fork(sw_out[i], fork_out[i]);
#endif
        single_layer_conv1_Conv2D_coarse_out_loop: for(unsigned int j=0;j<SINGLE_LAYER_CONV1_CONV2D_COARSE_OUT;j++) {
            #pragma HLS unroll
            single_layer_conv1_Conv2D_conv(weights[i][j], fork_out[i][j], conv_out[i][j]);
#if SINGLE_LAYER_CONV1_CONV2D_ACCUM_CHANNELS > 1
            single_layer_conv1_Conv2D_accum(conv_out[i][j], accum_out[i][j]);
#endif
        }
    }

#if SINGLE_LAYER_CONV1_CONV2D_ACCUM_CHANNELS > 1
#if SINGLE_LAYER_CONV1_CONV2D_HAS_BIAS == 1

    single_layer_conv1_Conv2D_glue(accum_out, glue_out);

    single_layer_conv1_Conv2D_coarse_out_bias_loop: for(unsigned int i=0;i<SINGLE_LAYER_CONV1_CONV2D_COARSE_OUT;i++) {
        #pragma HLS unroll
        single_layer_conv1_Conv2D_bias(biases[i], glue_out[i], out[i]);
    }

#else

    single_layer_conv1_Conv2D_glue(accum_out, out);

#endif
#else
#if SINGLE_LAYER_CONV1_CONV2D_HAS_BIAS == 1

    single_layer_conv1_Conv2D_glue(conv_out, glue_out);

    single_layer_conv1_Conv2D_coarse_out_bias_loop: for(unsigned int i=0;i<SINGLE_LAYER_CONV1_CONV2D_COARSE_OUT;i++) {
        #pragma HLS unroll
        single_layer_conv1_Conv2D_bias(biases[i], glue_out[i], out[i]);
    }

#else

    single_layer_conv1_Conv2D_glue(conv_out, out);

#endif
#endif

}

