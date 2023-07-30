<project xmlns="com.autoesl.autopilot.project" name="partition_0" top="fpgaconvnet_ip">
    <includePaths/>
    <libraryPaths/>
    <Simulation argv="">
        <SimFlow name="csim" ldflags="" mflags="" csimMode="0" lastCsimMode="0"/>
    </Simulation>
    <files xmlns="">
        <file name="../tb/single_layer_tb.cpp" sc="0" tb="1" cflags=" -I../include -I../data -IC:/Users/11569/.conda/envs/fpgaconvnet/Lib/site-packages/fpgaconvnet/hls/hardware  -IC:/Users/11569/.conda/envs/fpgaconvnet/Lib/site-packages/fpgaconvnet/hls/hardware/hlslib/include  -std=c++11 -fexceptions -Wno-unknown-pragmas" csimflags=" -Wno-unknown-pragmas" blackbox="false"/>
        <file name="../data/squeeze_single_layer_conv1_Relu_0.dat" sc="0" tb="1" cflags=" -Wno-unknown-pragmas" csimflags=" -Wno-unknown-pragmas" blackbox="false"/>
        <file name="../data/single_layer_conv1_Conv2D_weights_0.dat" sc="0" tb="1" cflags=" -Wno-unknown-pragmas" csimflags=" -Wno-unknown-pragmas" blackbox="false"/>
        <file name="../data/single_layer_conv1_Conv2D_0.dat" sc="0" tb="1" cflags=" -Wno-unknown-pragmas" csimflags=" -Wno-unknown-pragmas" blackbox="false"/>
        <file name="partition_0/src/squeeze_single_layer_conv1_Relu.cpp" sc="0" tb="false" cflags="-std=c++11 -fexceptions -Ipartition_0/include -Ipartition_0/data -IC:/Users/11569/.conda/envs/fpgaconvnet/Lib/site-packages/fpgaconvnet/hls/hardware -IC:/Users/11569/.conda/envs/fpgaconvnet/Lib/site-packages/fpgaconvnet/hls/hardware/hlslib/include" csimflags="" blackbox="false"/>
        <file name="partition_0/src/single_layer_top.cpp" sc="0" tb="false" cflags="-std=c++11 -fexceptions -Ipartition_0/include -Ipartition_0/data -IC:/Users/11569/.conda/envs/fpgaconvnet/Lib/site-packages/fpgaconvnet/hls/hardware -IC:/Users/11569/.conda/envs/fpgaconvnet/Lib/site-packages/fpgaconvnet/hls/hardware/hlslib/include" csimflags="" blackbox="false"/>
        <file name="partition_0/src/single_layer_conv1_Relu.cpp" sc="0" tb="false" cflags="-std=c++11 -fexceptions -Ipartition_0/include -Ipartition_0/data -IC:/Users/11569/.conda/envs/fpgaconvnet/Lib/site-packages/fpgaconvnet/hls/hardware -IC:/Users/11569/.conda/envs/fpgaconvnet/Lib/site-packages/fpgaconvnet/hls/hardware/hlslib/include" csimflags="" blackbox="false"/>
        <file name="partition_0/src/single_layer_conv1_Conv2D.cpp" sc="0" tb="false" cflags="-std=c++11 -fexceptions -Ipartition_0/include -Ipartition_0/data -IC:/Users/11569/.conda/envs/fpgaconvnet/Lib/site-packages/fpgaconvnet/hls/hardware -IC:/Users/11569/.conda/envs/fpgaconvnet/Lib/site-packages/fpgaconvnet/hls/hardware/hlslib/include" csimflags="" blackbox="false"/>
    </files>
    <solutions xmlns="">
        <solution name="solution" status="active"/>
    </solutions>
</project>

