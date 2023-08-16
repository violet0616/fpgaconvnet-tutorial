import numpy as np
import onnxruntime
import serial


# Grabs image at index idx from the MNIST testing dataset
def get_MNIST_image(idx):
    # Read the mnist binary format
    image_path = "MNIST/t10k-images.idx3-ubyte"
    with open(image_path, 'rb') as fh:
        fh.seek(16 + idx * (28 * 28), 0)
        arr = list(fh.read(28 * 28))
    arr = np.array(arr)
    arr = arr.astype('float32')

    # Normalize array [-1, 1]
    arr = np.subtract(arr, (np.amax(arr) + np.amin(arr)) / 2)
    arr = np.multiply(arr, 1 / np.amax(arr))

    arr = np.reshape(arr, (28, 28))
    arr.transpose()

    # Batches for onnx run
    arr = np.reshape(arr, (1, 1, 28, 28))
    arr = np.tile(arr, (42, 1, 1, 1))
    return arr

def get_CIFAR_image(idx):
    # Read the mnist binary format
    image_path = "MNIST/t10k-images.idx3-ubyte"
    with open(image_path, 'rb') as fh:
        fh.seek(16 + idx * (28 * 28), 0)
        arr = list(fh.read(28 * 28))
    arr = np.array(arr)
    arr = arr.astype('float32')

    # Normalize array [-1, 1]
    arr = np.subtract(arr, (np.amax(arr) + np.amin(arr)) / 2)
    arr = np.multiply(arr, 1 / np.amax(arr))

    # Resize to (32, 32)
    arr = np.reshape(arr, (28, 28))
    arr = np.pad(arr, ((2, 2), (2, 2)), mode='constant')

    # Batches for onnx run
    arr = np.reshape(arr, (1, 1, 32, 32))
    arr = np.tile(arr, (42, 1, 1, 1))
    return arr

# Grabs corresponding label at index idx from the MNIST testing dataset
def get_MNIST_label(idx):
    label_path = "MNIST/t10k-labels.idx1-ubyte"
    with open(label_path, 'rb') as fh:
        fh.seek(8 + idx, 0)
        return int(fh.read(1)[0])

# Runs inference using the onnxruntime
def run_inference(model_path, input):
    ort_sess = onnxruntime.InferenceSession(model_path)
    return ort_sess.run(["conv1"], {"conv1_input": input})


def run_inference_unknown_name(ort_sess, input):
    output_names = [output.name for output in ort_sess.get_outputs()]  # Get actual output names from the model
    return ort_sess.run(None, {"conv1_input": input})


# Sends 8 bit grayscale image to FPGA
def send_array(serial_descriptor, arr):
    # convert to 16 bit (8 fractional bit) fixed point format
    arr = np.multiply(arr, 256)
    arr = arr.astype('int32')

    # Serialize to space delimited ascii string
    arr = arr.flatten('C')  #C按照行来, F按照列
    arrstr = ""             #初始字符串
    new_line= "\n"
    arrstr1 = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 1 0 0 1 0 0 2 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1 2 3 1 7 16 15 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 13 22 22 22 21 22 22 22 4 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 6 18 22 21 21 22 22 22 21 17 17 5 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 19 22 21 22 21 22 22 22 22 22 21 19 4 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 6 20 22 22 21 10 3 4 9 22 22 21 11 2 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 5 9 10 10 3 0 0 0 17 21 22 22 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1 0 6 21 22 22 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 3 22 22 22 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 1 1 9 22 22 21 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 1 16 21 22 21 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 4 8 7 0 0 2 21 21 22 18 0 1 0 0 0 0 0 0 0 0 0 1 1 2 9 9 16 16 20 22 22 16 16 20 21 22 18 2 1 0 0 0 0 0 0 0 0 0 3 9 16 21 22 21 22 21 22 22 21 22 22 21 22 22 19 11 1 0 0 0 0 0 0 0 0 0 11 22 20 22 22 21 20 18 21 22 22 22 21 22 22 21 22 21 15 1 0 0 0 0 0 1 0 1 22 22 22 20 17 7 0 5 17 21 22 20 21 12 9 17 22 21 22 16 0 0 0 0 1 0 2 0 21 21 22 22 19 20 19 18 21 22 22 21 7 0 0 2 16 21 21 21 0 0 0 0 0 0 0 0 14 22 22 21 22 21 22 22 22 17 9 5 0 2 0 1 0 5 22 14 0 0 0 0 0 2 0 0 1 3 14 17 15 16 14 4 0 1 1 1 0 0 0 0 0 0 1 0 0 0 0 0 1 0 1 0 1 0 0 0 0 0 1 0 1 0 0 0 1 0 1 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 2 0 0 1 0 0 1 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
    for pixel in arr:
        arrstr += str(pixel) + " "

    # End string in newline
    arrstr = (arrstr[:-1]).encode('ascii', 'replace')
    arrstr1 = (arrstr1[:-1]).encode('ascii', 'replace')
    
    # Send over serial
    print(arrstr)
    # print(arrstr1)
    with serial.Serial(port=serial_descriptor, baudrate=115200, bytesize=8, parity='N', stopbits=1, timeout=None,
                       xonxoff=0,
                       rtscts=0) as ser:
        ser.write(arrstr)
    # Send a new line character to end the std::getline in C++
    with serial.Serial(port=serial_descriptor, baudrate=115200, bytesize=8, parity='N', stopbits=1, timeout=None,
                       xonxoff=0,
                       rtscts=0) as ser:
        ser.write(new_line.encode())
    return

# Sends 8 bit grayscale image to FPGA
def send_array_Haffmancoding(serial_descriptor, arrstr1):
    new_line= "\n"
    arrstr1 = (arrstr1[:-1]).encode('ascii', 'replace')
    
    # Send over serial
    print(arrstr1)
    with serial.Serial(port=serial_descriptor, baudrate=115200, bytesize=8, parity='N', stopbits=1, timeout=None,
                       xonxoff=0,
                       rtscts=0) as ser:
        ser.write(arrstr1)
    # Send a new line character to end the std::getline in C++
    with serial.Serial(port=serial_descriptor, baudrate=115200, bytesize=8, parity='N', stopbits=1, timeout=None,
                       xonxoff=0,
                       rtscts=0) as ser:
        ser.write(new_line.encode())
    return


#Decodes received string from FPGA as array
def receive_array(serial_descriptor):
    with serial.Serial(port=serial_descriptor, baudrate=115200, bytesize=8, parity='N', stopbits=1, timeout=None,
                       xonxoff=0,
                       rtscts=0) as ser:
        line = ser.readline()

    flat_fpgaoutput = list(map(int, line.split()))
    flat_fpgaoutput = np.array(flat_fpgaoutput).astype('float32')
    flat_fpgaoutput = np.divide(flat_fpgaoutput, 256)
    return flat_fpgaoutput

#Returns line received from FPGA
def receive_string(serial_descriptor):
    with serial.Serial(port=serial_descriptor, baudrate=115200, bytesize=8, parity='N', stopbits=1, timeout=None,
                       xonxoff=0, rtscts=0) as ser:
        return ser.readline().decode('ascii')
