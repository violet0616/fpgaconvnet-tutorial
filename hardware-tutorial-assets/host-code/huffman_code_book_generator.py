from IPython.display import Image 
import onnx
from onnx import helper, numpy_helper
import onnxruntime
import numpy as np 

from matplotlib import pyplot as plt
import tutorial_library
from collections import Counter
from HuffmanNode import *

total_layer_number = 7 
The_place_of_Huff_where_you_want_to_insert = 1





def main():

    model_path = "C:/IC/2023_project/2023_project/FpgaConvnet_forked/fpgaconvnet-tutorial/models/mpcnn.onnx"  # 替换为你的模型文件路径
    model0_path = "C:/IC/2023_project/2023_project/FpgaConvnet_forked/fpgaconvnet-tutorial/models/model0.onnx"  # 替换为你的模型文件路径
    model1_path = "C:/IC/2023_project/2023_project/FpgaConvnet_forked/fpgaconvnet-tutorial/models/model1.onnx"  # 替换为你的模型文件路径
    model2_path = "C:/IC/2023_project/2023_project/FpgaConvnet_forked/fpgaconvnet-tutorial/models/model2.onnx"  # 替换为你的模型文件路径
    model3_path = "C:/IC/2023_project/2023_project/FpgaConvnet_forked/fpgaconvnet-tutorial/models/model3.onnx"  # 替换为你的模型文件路径
    model4_path = "C:/IC/2023_project/2023_project/FpgaConvnet_forked/fpgaconvnet-tutorial/models/model4.onnx"  # 替换为你的模型文件路径
    model5_path = "C:/IC/2023_project/2023_project/FpgaConvnet_forked/fpgaconvnet-tutorial/models/model5.onnx"  # 替换为你的模型文件路径
    model6_path = "C:/IC/2023_project/2023_project/FpgaConvnet_forked/fpgaconvnet-tutorial/models/model6.onnx"  # 替换为你的模型文件路径
    model7_path = "C:/IC/2023_project/2023_project/FpgaConvnet_forked/fpgaconvnet-tutorial/models/model7.onnx"  # 替换为你的模型文件路径


    model_paths = [model0_path, model1_path, model2_path, model3_path, model4_path, model5_path, model6_path, model7_path]

 
    # nm_image_data_base = 1
    # data_list = []
    # for i in range (nm_image_data_base):#collecting all possible input data. 
    #             mnist_idx = int(i)
                
    #             # Grab image from dataset
    #             mnist_image = tutorial_library.get_MNIST_image(mnist_idx)
    #             data_list.append(mnist_image)
                
    #             one_mnist_image = mnist_image[0][0]#mnist_image[0][0] to [42][0] represent same picture




    # Uncomment the folowing section of code for getting minst data set that is padded to 32* 32  for single_layer.onnx model.#############################################################################################################################
    mnist_idx = 10
    # Prepare input data for model single_layer.onnx
    mnist_image = tutorial_library.get_CIFAR_image_fake(mnist_idx)
    #Optional input featuremap display
    print(tutorial_library.get_MNIST_label(mnist_idx))
    # Display image
    print(mnist_image[0][0].shape)
    plt.imshow(mnist_image[0][0], interpolation='nearest')
    plt.show()

    
    # # Uncomment the folowing section of code for getting CIFAR data set for mpcnn.onnx model.#############################################################################################################################
    # CIFAR10_idx = 40000
    # # Prepare input data for model mpcnn.onnx
    # cifar_image = tutorial_library.get_CIFAR10_image(CIFAR10_idx)    # Transpose the image for proper display
    # transposed_image = cifar_image[0].transpose(1, 2, 0)  # Transpose to (32, 32, 3)
    # plt.imshow(transposed_image, interpolation='nearest')
    # plt.show()

    
    # Load the ONNX model
    model = onnx.load(model_path)
    layer_output_list = []
    onnx_model_list = []
    
    
    frequncy_map = []
    Huffmand_tree_root = []
    huffman_codes_maps =[]
    model_list = separate_model(model, total_layer_number, model_paths)

    # Gwt output list 
    for i in range (total_layer_number):
        onnx_model_list.append(onnxruntime.InferenceSession(model_paths[i]) )
        layer_output_list.append(tutorial_library.run_inference_unknown_name(onnx_model_list[i], mnist_image))

    
#   # Uncomment for qutising to specif number of binary bits    
#     num_bits = 10  
#     quantization_range = 2 ** num_bits - 1
#     quantized_layer_output_list = []
#     for layer_output in layer_output_list:
#         # Normalize the values to the range [0, 1]
#         normalized_values = (layer_output - np.min(layer_output)) / (np.max(layer_output) - np.min(layer_output))        
#         # Quantize the normalized values to the specified number of bits
#         quantized_values = np.round(normalized_values * quantization_range)
        
#         # Denormalize the quantized values back to the original range
#         denormalized_values = (quantized_values / quantization_range) * (np.max(layer_output) - np.min(layer_output)) + np.min(layer_output)
        
#         # Append the quantized values to the list
#         quantized_layer_output_list.append(denormalized_values)

    
# #    # Uncomment for qutising to less discrete level
#     num_bins = 256  # Number of discrete levels (adjust as needed)
#     quantized_layer_output_list = []
#     for layer_output in layer_output_list:
#          # normalized_values = (layer_output - np.min(layer_output)) / (np.max(layer_output) - np.min(layer_output))
#         flat_layer_output = np.array(layer_output).flatten()
#         # Step 1: Normalize data
#         normalized_data = (np.array(flat_layer_output) - np.min(flat_layer_output)) / (np.max(flat_layer_output) - np.min(flat_layer_output))
#         # Step 2: Quantization
#         quantized_data = np.round(normalized_data * (num_bins - 1)) / (num_bins - 1)
        
#         # quantized_data = np.digitize(flat_layer_output, np.linspace(min(flat_layer_output), max(flat_layer_output), num_bins))
#         quantized_layer_output_list.append(quantized_data)
    
    
    
    frequncy_map = analyze_data(layer_output_list[The_place_of_Huff_where_you_want_to_insert])
    # frequncy_map = analyze_data(quantized_layer_output_list[The_place_of_Huff_where_you_want_to_insert])
    
    
    # smoothed_data = {key: np.exp(value / 100) for key, value in frequncy_map.items()}
    # # Sort the dictionary by values
    # sorted_data = sorted(smoothed_data.items(), key=lambda item: item[1])
    # # Extract sorted x-values and corresponding y-values
    # x_values = [i for i, _ in enumerate(sorted_data)]
    # y_values = [value for _, value in sorted_data]
    # # Create a line plot
    # plt.plot(x_values, y_values, marker='o', linestyle='-', color='blue')
    # plt.xlabel('Sorted Index')
    # plt.ylabel('Value')
    # plt.title('Increase of Values')
    # # Show the plot
    # plt.show()



# # Uncomment for a curved frequncy map, not the optimised transition but better number of bits we hope
#     # Randomly pick transition points from the dictionary keys
#     num_transitions = 10  # Number of transition points to pick
#     all_keys = list(frequncy_map.keys())
#     transition_points = np.random.choice(all_keys, num_transitions, replace=False)
#     transition_points.sort()
#     # Interpolate values between randomly selected transition points
#     smoothed_data = {}
#     for key in frequncy_map:
#         smoothed_data[key] = frequncy_map[key]
#         if key in transition_points:
#             next_indices = np.where(transition_points == key)[0] + 1
#             if len(next_indices) > 0:
#                 next_index = next_indices[0]
#                 if next_index < len(transition_points):
#                     next_key = transition_points[next_index]
#                     steps = next_key - key
#                     diff = frequncy_map[next_key] - frequncy_map[key]
#                     for i in range(1, int(steps)):
#                         interpolated_value = frequncy_map[key] + (i / steps) * diff
#                         smoothed_data[key + i / 1000] = interpolated_value

#     # Plot the smoothed data
#     sorted_data = sorted(smoothed_data.items(), key=lambda item: item[1])
#     # Extract sorted x-values and corresponding y-values
#     x_values = [i for i, _ in enumerate(sorted_data)]
#     y_values = [value for _, value in sorted_data]
#     # Create a line plot
#     plt.plot(x_values, y_values, marker='o', linestyle='-', color='blue')
#     plt.xlabel('Sorted Index')
#     plt.ylabel('Value')
#     plt.title('Increase of Values')
#     # Show the plot
#     plt.show()








    # Building the Huffman tree and finding each node in the process
    Huffmand_tree_root = build_huffman_tree(frequncy_map)

    # Generate Huffman code map 
    huffman_codes_map = Huffman_tree_coding(Huffmand_tree_root)
    
    # Open a text file for writing
    with open(f"Huffman_codemap_{The_place_of_Huff_where_you_want_to_insert}.txt", "w") as output_file:
        print_out_codemap_to_txt(huffman_codes_map, file=output_file)




    # frequncy_map_2 = analyze_data(layer_output_list[2])
    # # Building the Huffman tree and finding each node in the process
    # Huffmand_tree_root_2 = build_huffman_tree(frequncy_map_2)
    # # Generate Huffman code map 
    # huffman_codes_map_2 = Huffman_tree_coding(Huffmand_tree_root_2)
    
    # frequncy_map_3 = analyze_data(layer_output_list[3])
    # # Building the Huffman tree and finding each node in the process
    # Huffmand_tree_root_3 = build_huffman_tree(frequncy_map_3)
    # # Generate Huffman code map 
    # huffman_codes_map_3 = Huffman_tree_coding(Huffmand_tree_root_3)
    
    # for output_data in layer_output_list:
    #     # Analyze the data and get the frequency map
    #     frequency_map = analyze_data(output_data)
        
    #     # Building the Huffman tree and finding each node in the process
    #     huffman_tree_root = build_huffman_tree(frequency_map)
        
    #     # Generate Huffman code map
    #     huffman_codes_map = Huffman_tree_coding(huffman_tree_root)
        
    #     # Append the Huffman code map to the list
    #     huffman_codes_maps.append(huffman_codes_map)







    # # Create a new model with only the first layer
    # new_model = onnx.ModelProto()
    # new_model.graph.CopyFrom(model.graph)

    # # Specify the ONNX OperatorSet version
    # opset_version = 13  
    # # Add the OperatorSet version to the model
    # new_model.opset_import.append(onnx.helper.make_opsetid("", opset_version))
    # new_model.ir_version = 7  # Set version

    # # Remove all nodes except the first node (ConvolutionLayer1)
    # new_model.graph.ClearField("node")
    # new_model.graph.node.extend([model.graph.node[0]])

    # # Remove all outputs except the output from the first node
    # new_model.graph.ClearField("output")
    # output_name = model.graph.node[0].output[0]
    # output_info = onnx.ValueInfoProto(name=output_name)
    # new_model.graph.output.extend([output_info])

    # # Update the ONNX model
    # onnx.save_model(new_model, model1_path) 

    
    # onnx_model_1 = onnxruntime.InferenceSession(model1_path) 
    # # layer_1_output = onnx_model_1.run([onnx_model_1.get_outputs()[0].name], {onnx_model_1.get_inputs()[0].name: mnist_image})
    # layer_1_output = tutorial_library.run_inference_unknown_name(onnx_model_1, mnist_image)

    return





def separate_model(model, total_layer_number, model_paths):
    """
    separate the model to read all 7 layers' outputs
    """
    model_list = []
    for i in range (total_layer_number):
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

# def get_freq_map_for_MNIST_images (nm_image_data_base,data_list):
#     """
#     Builds the freq
#     """
#     for i in range (nm_image_data_base):#collecting all possible input data. 
#             mnist_idx = int(i)
            
#             # Grab image from dataset
#             mnist_image = tutorial_library.get_MNIST_image(mnist_idx)
#             data_list.append(mnist_image)
            
#             one_mnist_image = mnist_image[0][0]#mnist_image[0][0] to [42][0] represent same picture
        
#     frequncy_map = analyze_data(data_list)
#     return frequncy_map   

if __name__ == '__main__':
    main()
    