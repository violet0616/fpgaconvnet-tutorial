import numpy as np
import tutorial_library
from matplotlib import pyplot as plt
import sys
from collections import Counter
from HuffmanNode import HuffmanNode
import heapq
        
def main():

    # #Simple command line parsing
    # fpga_serial = sys.argv[1]
    fpga_serial = "COM4"
    
    # #Optional third argument
    # if len(sys.argv) == 2:
    #     mnist_idx = 0
    # elif len(sys.argv) > 2:
    #     mnist_idx = int(sys.argv[2])
    
    # # Haffman coding process for minist_image_data_set###################################################################################################
    # hls_test_string = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 1 0 0 1 0 0 2 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1 2 3 1 7 16 15 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 13 22 22 22 21 22 22 22 4 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 6 18 22 21 21 22 22 22 21 17 17 5 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 19 22 21 22 21 22 22 22 22 22 21 19 4 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 6 20 22 22 21 10 3 4 9 22 22 21 11 2 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 5 9 10 10 3 0 0 0 17 21 22 22 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1 0 6 21 22 22 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 3 22 22 22 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 1 1 9 22 22 21 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 1 16 21 22 21 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 4 8 7 0 0 2 21 21 22 18 0 1 0 0 0 0 0 0 0 0 0 1 1 2 9 9 16 16 20 22 22 16 16 20 21 22 18 2 1 0 0 0 0 0 0 0 0 0 3 9 16 21 22 21 22 21 22 22 21 22 22 21 22 22 19 11 1 0 0 0 0 0 0 0 0 0 11 22 20 22 22 21 20 18 21 22 22 22 21 22 22 21 22 21 15 1 0 0 0 0 0 1 0 1 22 22 22 20 17 7 0 5 17 21 22 20 21 12 9 17 22 21 22 16 0 0 0 0 1 0 2 0 21 21 22 22 19 20 19 18 21 22 22 21 7 0 0 2 16 21 21 21 0 0 0 0 0 0 0 0 14 22 22 21 22 21 22 22 22 17 9 5 0 2 0 1 0 5 22 14 0 0 0 0 0 2 0 0 1 3 14 17 15 16 14 4 0 1 1 1 0 0 0 0 0 0 1 0 0 0 0 0 1 0 1 0 1 0 0 0 0 0 1 0 1 0 0 0 1 0 1 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 2 0 0 1 0 0 1 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
    # string_freq_map = get_freq_map_for_strings(hls_test_string)
    # test_tree_root = build_huffman_tree(string_freq_map)
    # test_codes_map = Huffman_tree_coding(test_tree_root)
    # hls_test_array = np.array([float(x) for x in hls_test_string.split()])
    # coded_string= array_to_Haffman_string(hls_test_array,test_codes_map)
    
    # total_bits_bef_comp = 32*len(coded_string.split()) #32 bits per data before compression
    # total_bits_aft_comp = counting_total_bits_after_comp(coded_string)
    # # FPGA upload
    # print("Beginning featuremap upload")
    # print(f"Total number of bits that need to be transmitted before compressing:{total_bits_bef_comp}")
    # print(f"Total number of bits that need to be transmitted after compressing:{total_bits_aft_comp}")
    # tutorial_library.send_array_Haffmancoding(fpga_serial, coded_string)

  
    
    
    
    # # Haffman coding process for minist_image_data_set###################################################################################################
    # mnist_image_list = []   
    # freq_map = get_freq_map_for_MNIST_images(1, mnist_image_list)
    # # Building the Huffman tree and finding each node in the process
    # Huffmand_tree_root = build_huffman_tree(freq_map)
    # # Generate Huffman code map 
    # huffman_codes_map = Huffman_tree_coding(Huffmand_tree_root)
    # # compressing the data that needs to be tranmitted for one of the image
    # coded_string= array_to_Haffman_string(mnist_image_list[0][0][0],huffman_codes_map)    
    # # Optional input featuremap display
    # # print(tutorial_library.get_MNIST_label(mnist_idx))
    # # Display image
    # plt.imshow(mnist_image_list[0][0][0], interpolation='nearest')
    # plt.show()
    
    # # curve
    # total_bits_bef_comp = 32*len(mnist_image_list[0][0][0].flatten('C')) #32 bits per data before compression
    # total_bits_aft_comp = counting_total_bits_after_comp(coded_string)
    # # FPGA upload
    # print("Beginning featuremap upload")
    # print(f"Total number of bits that need to be transmitted for 1 mnist image before compressing:{total_bits_bef_comp}")
    # print(f"Total number of bits that need to be transmitted for 1 mnist image after compressing:{total_bits_aft_comp}")
    # tutorial_library.send_array_Haffmancoding(fpga_serial, coded_string)



    # Confirm featuremap reception / additional status messages
    print(tutorial_library.receive_string(fpga_serial))
    print(tutorial_library.receive_string(fpga_serial))
    flat_fpgaoutput = tutorial_library.receive_array(fpga_serial)
    print(tutorial_library.receive_string(fpga_serial))

    # Reference ONNX computations
    output = tutorial_library.run_inference("models/single_layer.onnx", mnist_image)
    output = output[0][0]

    #Flattening for array preview
    fpgaoutput = np.reshape(flat_fpgaoutput, (24, 24, 16))
    flat_output = output.flatten()

    #Array comparison print
    print(f"Flat FPGA readout ({fpgaoutput.size} items):")
    print(flat_fpgaoutput)
    print(f"Flat Reference readout ({output.size} items):")
    print(flat_output)

    #Error information
    err = (np.square(fpgaoutput - output)).mean(axis=None)
    print(f"The mean squared error was: {err}")
    return
    
def analyze_data(data):
    """
    Analyzes the input data and returns a dictionary containing the frequency
    of occurrence for each unique symbol (16-bit data).

    Parameters:
        data (numpy.ndarray): Input data as a 2D numpy array.

    Returns:
        dict: A dictionary containing the frequency of occurrence for each unique symbol.
    """
    data = np.array(data)
    # Flatten the 2D data array to a 1D array
    flattened_data = data.flatten()

    # Count the frequency of each unique symbol using the Counter class
    symbol_frequencies = Counter(flattened_data)

    return symbol_frequencies

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

def get_freq_map_for_strings (stra):
    integer_list = [int(x) for x in stra.split()]
   
    frequncy_map = analyze_data(integer_list)
    return frequncy_map  

def build_huffman_tree(frequency_map):
    """
    Builds the Huffman tree based on the frequency map.

    Parameters:
        frequency_map (dict): A dictionary containing the frequency of occurrence for each unique symbol.

    Returns:
        HuffmanNode: The root node of the Huffman tree.
    """
    # node is a list of HuffmanNode objects, with (symbol and freq )
    nodes = [HuffmanNode(symbol, frequency) for symbol, frequency in frequency_map.items()]

    # Convert the list into a priority queue (min heap) based on frequency
    sorted_nodes = sorted(nodes, key=lambda x: x.frequency)
    
    while len(sorted_nodes) > 1:
        # Extract the two nodes with the lowest frequencies
        left_child = sorted_nodes.pop(0)
        right_child = sorted_nodes.pop(0)

        # Create a new parent node with combined frequency
        parent_frequency = left_child.frequency + right_child.frequency
        parent_node = HuffmanNode(None, parent_frequency)
        parent_node.left = left_child
        parent_node.right = right_child

        # Add the parent node back to the heap
        sorted_nodes.append(parent_node)
        sorted_nodes = sorted(sorted_nodes, key=lambda x: x.frequency)

    # The remaining node in the heap is the root of the Huffman tree
    huffman_tree_root = sorted_nodes[0]

    return huffman_tree_root
        
def Huffman_tree_coding(root_node, current_code="", code_map={}):
    if root_node is None:
        return

    # If the node is a leaf (i.e., has a symbol), store the Huffman code in the code map
    if root_node.symbol is not None:
        code_map[root_node.symbol] = current_code

    Huffman_tree_coding(root_node.left, current_code + "0", code_map)
    Huffman_tree_coding(root_node.right, current_code + "1", code_map)

    return code_map

def array_to_Haffman_string(array_in,code_map):
    string_H = ""
    array_in = array_in.flatten('C')  #C按照行来, F按照列

    for element in array_in:
        element = float(element)
        for symbol,code in code_map.items():
            if element == symbol:
                
                string_H += code + " "
    return string_H

def counting_total_bits_after_comp(string_in):
        count_0 = string_in.count('0')
        count_1 = string_in.count('1')   
        return count_0+count_1 
    
    
if __name__ == '__main__':
    main()
    
    
