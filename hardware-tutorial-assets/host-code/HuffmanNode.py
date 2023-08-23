import numpy as np
import tutorial_library
from matplotlib import pyplot as plt
from collections import Counter
import copy
import sys

class HuffmanNode:
    def __init__(self, symbol, frequency):
        self.symbol = symbol
        self.frequency = frequency
        self.left = None
        self.right = None


def analyze_data(data):
    """
    Analyzes the input data and returns a dictionary containing the frequency
    of occurrence for each unique symbol (16-bit data).

    Parameters:
        data (numpy.ndarray): Input data as a 2D numpy array.

    Returns:
        dict: A dictionary containing the frequency of occurrence for each unique symbol.
    """
    data_copy = copy.deepcopy(data)
    data_copy = np.array(data_copy)
    # Flatten the 2D data array to a 1D array
    flattened_data = data_copy.flatten()

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
    
    
def print_out_codemap(code_map):
    print("{", end="")
    for key, value in code_map.items():
        
        print(f"{{0b{value}, {len(value)}, {key}}}", end="")
        if key != list(code_map.keys())[-1]:
            print(", ", end="")
    print("}")

def print_out_codemap_to_txt(code_map, file=sys.stdout):
    print("{", end="", file=file)
    for key, value in code_map.items():
        print(f"{{0b{value}, {len(value)}, {key}}}", end="", file=file)
        if key != list(code_map.keys())[-1]:
            print(", ", end="", file=file)
    print("}", file=file)
    
def print_out_padding_codemap(code_map, padded_bits):
    print("{", end="")
    for key, value in code_map.items():
        padded_value = "1" * (padded_bits - len(value)) + value
        print(f"{{0b{padded_value}, {key}}}", end="")
        if key != list(code_map.keys())[-1]:
            print(", ", end="")
    print("}")