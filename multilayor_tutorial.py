from IPython.display import Image 
from fpgaconvnet.models.network import Network

# load network
net = Network("3_layers", "models/mpcnn.onnx")
net1 = Network("Parti_1", "models/mpcnn.onnx")
net2 = Network("Parti_2", "models/mpcnn.onnx")
net3 = Network("Parti_3", "models/mpcnn.onnx")
#net.visualise("baseline.png", mode="png")

# display to jupyter notebook
im = Image('baseline.png')
#display(im)

# load the zedboard platform details
net.update_platform("platforms/zedboard.json")
net1.update_platform("platforms/zedboard.json")
net2.update_platform("platforms/zedboard.json")
net3.update_platform("platforms/zedboard.json")
# show latency, throughput and resource predictions
print(f"predicted latency (us): {net.get_latency()*1000000}")
print(f"predicted throughput (img/s): {net.get_throughput()} (batch size={net.batch_size})")
print(f"predicted resource usage: {net.partitions[0].get_resource_usage()}")
#!mkdir -p outputs
import sys
sys.path.append('C:\\IC\\2023_project\\2023_project\\FpgaConvnet\\fpgaconvnet-tutorial\\samo\\samo')  # Add the path to the folder containing the cli script

from cli import main_for_multi  # Import the main function from the cli script
from cli import main_for_one

# invoking the CLI from python
# main_for_one([
#     "--model", "models/mpcnn.onnx",
#     "--platform", "platforms/zedboard.json",
#     "--output-path", "outputs/mpcnn_opt.json",
#     "--backend", "fpgaconvnet",
#     "--optimiser", "rule",
#     "--objective", "latency",
#     "--enable_reconf", "true"
# ])

# # load the optimised network
# net.load_network("outputs/mpcnn_opt.json") # TODO: change name
# net.update_partitions()

# # print the performance and resource predictions
# print(f"predicted latency (us): {net.get_latency()*1000000}")
# print(f"predicted throughput (img/s): {net.get_throughput()} (batch size={net.batch_size})")
# print(f"predicted resource usage: {net.partitions[0].get_resource_usage()}")



# multipart
main_for_multi([
    "--model", "models/mpcnn.onnx",
    "--model1", "models/mpcnn1.onnx",
    "--model2", "models/mpcnn2.onnx",
    "--model3", "models/mpcnn3.onnx",
    "--platform", "platforms/zedboard.json",
    "--output-path", "outputs/mpcnn_opt.json",
    "--output-path1", "outputs/mpcnn_opt1.json",
    "--output-path2", "outputs/mpcnn_opt2.json",
    "--output-path3", "outputs/mpcnn_opt3.json",
    "--backend", "fpgaconvnet",
    "--optimiser", "rule",
    "--objective", "latency",
    "--enable_reconf", "true"
])

# load the optimised network
net1.load_network("outputs/mpcnn_opt1.json") # TODO: change name
net2.load_network("outputs/mpcnn_opt2.json") # TODO: change name
net3.load_network("outputs/mpcnn_opt3.json") # TODO: change name

net1.update_partitions()
net2.update_partitions()
net3.update_partitions()

lat_list = sorted [net1.get_latency()*1000000, net2.get_latency()*1000000, net3.get_latency()*1000000]
throu_list= sorted [net1.get_throughput(), net2.get_throughput(), net3.get_throughput()]

# print the performance and resource predictions
print(f"predicted latency (us): {lat_list[-1]}")
print(f"predicted throughput (img/s): {throu_list[0]} (batch size={net.batch_size})")
overall_throughput_improve = throu_list[0]/net.get_throughput()
print(f"The overall throuput has increased {overall_throughput_improve} times")


