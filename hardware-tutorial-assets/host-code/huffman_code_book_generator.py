from IPython.display import Image 
import onnx
from onnx import helper, numpy_helper
import onnxruntime
import numpy as np 

from matplotlib import pyplot as plt
import tutorial_library
from collections import Counter
from HuffmanNode import HuffmanNode

layer_number = 7 

def main():

    model_path = "C:/IC/2023_project/2023_project/FpgaConvnet_forked/fpgaconvnet-tutorial/models/mpcnn.onnx"  # 替换为你的模型文件路径
    model1_path = "C:/IC/2023_project/2023_project/FpgaConvnet_forked/fpgaconvnet-tutorial/models/model1.onnx"  # 替换为你的模型文件路径
    model2_path = "C:/IC/2023_project/2023_project/FpgaConvnet_forked/fpgaconvnet-tutorial/models/model2.onnx"  # 替换为你的模型文件路径
    model3_path = "C:/IC/2023_project/2023_project/FpgaConvnet_forked/fpgaconvnet-tutorial/models/model3.onnx"  # 替换为你的模型文件路径
    model4_path = "C:/IC/2023_project/2023_project/FpgaConvnet_forked/fpgaconvnet-tutorial/models/model4.onnx"  # 替换为你的模型文件路径
    model5_path = "C:/IC/2023_project/2023_project/FpgaConvnet_forked/fpgaconvnet-tutorial/models/model5.onnx"  # 替换为你的模型文件路径
    model6_path = "C:/IC/2023_project/2023_project/FpgaConvnet_forked/fpgaconvnet-tutorial/models/model6.onnx"  # 替换为你的模型文件路径
    model7_path = "C:/IC/2023_project/2023_project/FpgaConvnet_forked/fpgaconvnet-tutorial/models/model7.onnx"  # 替换为你的模型文件路径


    model_paths = model_paths = [model1_path, model2_path, model3_path, model4_path, model5_path, model6_path, model7_path]

    # onnx_modelion = onnxruntime.InferenceSession(model_path)

    # nm_image_data_base = 1
    # data_list = []
    # for i in range (nm_image_data_base):#collecting all possible input data. 
    #             mnist_idx = int(i)
                
    #             # Grab image from dataset
    #             mnist_image = tutorial_library.get_MNIST_image(mnist_idx)
    #             data_list.append(mnist_image)
                
    #             one_mnist_image = mnist_image[0][0]#mnist_image[0][0] to [42][0] represent same picture


    mnist_idx = 10
    # Prepare input data (replace with your own data)
    mnist_image = tutorial_library.get_CIFAR_image(mnist_idx)

    #Optional input featuremap display
    print(tutorial_library.get_MNIST_label(mnist_idx))
    # Display image
    print(mnist_image[0][0].shape)
    plt.imshow(mnist_image[0][0], interpolation='nearest')
    plt.show()



    

    
    # Load the ONNX model
    model = onnx.load(model_path)
    layer_output_list = []
    onnx_model_list = []
    
    model_list = separate_model(model, layer_number, model_paths)

    # onnx_model_1 = onnxruntime.InferenceSession(model_paths[0]) 
    # onnx_model_2 = onnxruntime.InferenceSession(model_paths[1]) 

    # # layer_1_output = onnx_model_1.run([onnx_model_1.get_outputs()[0].name], {onnx_model_1.get_inputs()[0].name: mnist_image})
    # layer_1_output = tutorial_library.run_inference_unknown_name(onnx_model_1, mnist_image)
    # layer_2_output = tutorial_library.run_inference_unknown_name(onnx_model_2, mnist_image)


    for i in range (layer_number):
        onnx_model_list.append(onnxruntime.InferenceSession(model_paths[i]) )
        layer_output_list.append(tutorial_library.run_inference_unknown_name(onnx_model_list[i], mnist_image))

    
        






    # Create a new model with only the first layer
    new_model = onnx.ModelProto()
    new_model.graph.CopyFrom(model.graph)

    # Specify the ONNX OperatorSet version
    opset_version = 13  
    # Add the OperatorSet version to the model
    new_model.opset_import.append(onnx.helper.make_opsetid("", opset_version))
    new_model.ir_version = 7  # Set version

    # Remove all nodes except the first node (ConvolutionLayer1)
    new_model.graph.ClearField("node")
    new_model.graph.node.extend([model.graph.node[0]])

    # Remove all outputs except the output from the first node
    new_model.graph.ClearField("output")
    output_name = model.graph.node[0].output[0]
    output_info = onnx.ValueInfoProto(name=output_name)
    new_model.graph.output.extend([output_info])

    # Update the ONNX model
    onnx.save_model(new_model, model1_path) 





    
    onnx_model_1 = onnxruntime.InferenceSession(model1_path) 
    # layer_1_output = onnx_model_1.run([onnx_model_1.get_outputs()[0].name], {onnx_model_1.get_inputs()[0].name: mnist_image})
    layer_1_output = tutorial_library.run_inference_unknown_name(onnx_model_1, mnist_image)


    a=1
    return





def separate_model(model, layer_number, model_paths):
    """
    separate the model to read all 7 layers' outputs
    """
    model_list = []
    for i in range (layer_number):
         # Create a new model with only the first layer
        new_model = onnx.ModelProto()
        new_model.graph.CopyFrom(model.graph)

        # Specify the ONNX OperatorSet version
        opset_version = 13  
        # Add the OperatorSet version to the model
        new_model.opset_import.append(onnx.helper.make_opsetid("", opset_version))
        new_model.ir_version = 7  # Set version
        # Remove all nodes except the first node (ConvolutionLayer1)
        new_model.graph.ClearField("node")
        
        for j in range (i+1):
            new_model.graph.node.extend([model.graph.node[j]])
        
        # Remove all outputs except the output from the first node
        new_model.graph.ClearField("output")
        output_name = model.graph.node[i].output[0]
        output_info = onnx.ValueInfoProto(name=output_name)
        new_model.graph.output.extend([output_info])
        
        # Update the ONNX model
        onnx.save_model(new_model, model_paths[i])    
        model_list.append(new_model)
        
    return model_list

def get_freq_map_for_MNIST_images (nm_image_data_base,data_list):
    """
    Builds the freq
    """
    for i in range (nm_image_data_base):#collecting all possible input data. 
            mnist_idx = int(i)
            
            # Grab image from dataset
            mnist_image = tutorial_library.get_MNIST_image(mnist_idx)
            data_list.append(mnist_image)
            
            one_mnist_image = mnist_image[0][0]#mnist_image[0][0] to [42][0] represent same picture
        
    frequncy_map = analyze_data(data_list)
    return frequncy_map   





if __name__ == '__main__':
    main()
    