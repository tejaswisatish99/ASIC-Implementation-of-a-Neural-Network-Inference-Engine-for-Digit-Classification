# Import necessary packages
# Data handling packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# Math handling packages
import numpy as np
# Graphing packages
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
# CSV file handling packages
import csv
# Misc packages
import sys
import os
import re
from IPython.display import Markdown as md
from IPython.display import Image as img
from IPython.display import HTML as html

%matplotlib inline

# Helper functions to read and print files in notebook, formatted with markdown
def openVerilogFile(title, file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        c = f.read()
        f.close()
    display(md(f"**{title}**"))
    display(md(f"``````"))

def openImage(title, file_path, width=600, height=600):
    display(md(f"**{title}**"))
    display(img(filename=file_path, width=width, height=height))

# Utility for exporting data to MEM/VH files for Verilog/FPGA
class SaveToFile:
    def __init__(self, width=8, signed=True, fmt="hex"):
        self.width = width
        self.signed = signed
        self.fmt = fmt
    def _ensure_dir(self, filename):
        folder = os.path.dirname(filename)
        if folder:
            os.makedirs(folder, exist_ok=True)
    def _int_to_str(self, num):
        if num < 0:
            num = (1 << self.width) + num
        num &= (1 << self.width) - 1
        if self.fmt == "hex":
            return f"{num:0{self.width//4}X}"
        elif self.fmt == "bin":
            return f"{num:0{self.width}b}"
        else:
            raise ValueError("fmt must be 'hex' or 'bin'")
    def _bits_to_str(self, bits):
        bitstr = "".join(str(b) for b in reversed(bits))
        if self.fmt == "hex":
            hex_width = (len(bits) + 3) // 4
            return f"{int(bitstr, 2):0{hex_width}X}"
        elif self.fmt == "bin":
            return bitstr
        else:
            raise ValueError("fmt must be 'hex' or 'bin'")
    def to_mem(self, arr, filename="out.mem", compile_bits=False):
        self._ensure_dir(filename)
        with open(filename, "w") as f:
            if compile_bits:
                for row in arr:
                    f.write(self._bits_to_str(row) + "\n")
            else:
                for val in arr.flatten():
                    f.write(self._int_to_str(int(val)) + "\n")
    def to_vh(self, arr, filename="out.vh", var_name="mem", compile_bits=False):
        self._ensure_dir(filename)
        with open(filename, "w") as f:
            if compile_bits:
                for i, row in enumerate(arr):
                    bitstr = "".join(str(b) for b in reversed(row))
                    f.write(f"assign {var_name}[{i}] = {len(row)}'b{bitstr};\n")
            else:
                for i, val in enumerate(arr.flatten()):
                    val_tc = self._int_to_str(int(val))
                    if self.fmt == "hex":
                        f.write(f"assign {var_name}[{i}] = {self.width}'h{val_tc};\n")
                    else:
                        prefix = f"{self.width}'{'s' if self.signed else ''}b"
                        f.write(f"assign {var_name}[{i}] = {prefix}{val_tc};\n")

def get_table(index = 0):
    with open("./metrics.tab") as m:
        mstream = m.read()
        m.close()
    metrics = mstream.split(';;;')
    return metrics[index].strip()

# Data loader/shuffler as used for training/testing
class load_data:
    def __init__(self, X, batch_size=32, shuffle_data=True):
        self.X = np.array(X)
        self.batch_size = batch_size
        self.shuffle_data = shuffle_data
    def __iter__(self):
        if self.shuffle_data:
            X = shuffle(self.X)
        else:
            X = self.X
        for start in range(0, len(X), self.batch_size):
            end = start + self.batch_size
            yield X[start:end]
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

def argmax_labels(labels: np.ndarray) -> np.ndarray:
    return np.argmax(labels, axis=1)

def read_csv(file_path):
    return pd.read_csv(file_path, header=0, delimiter=',')

data = read_csv('../corpus/semeion.csv')
c = data.iloc[:,:]
c = np.array(c)
(train, test) = train_test_split(c, test_size=0.2, random_state=10)
train_dataset = load_data(train, batch_size=1, shuffle_data=True)

print(f"Training Data Size: {train.shape[0]}x{train.shape[1]}")
print(f"Test Data Size: {test.shape[0]}x{test.shape[1]}")

testSave = SaveToFile(width=1, signed=False, fmt='bin')
testSave.to_mem(test[:,:256], "../model/test/test_ip.mem", compile_bits=True)
testSave = SaveToFile(width=4, signed=False, fmt='hex')
testSave.to_mem(argmax_labels(test[:,256:]), "../model/test/test_op.mem", compile_bits=False)

openImage("Implemented Design of NN", "../diagrams/png/nn_arch.png", 600, 600)

# Core NN functions as written in document
def forward_propagation(X, model):
    W1, b1, W2, b2 = model["W1"], model["b1"], model["W2"], model["b2"]
    Z1 = np.dot(W1, X) + b1
    A1 = leaky_relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = leaky_relu(Z2)
    return {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

def initialize_model(input_size, hidden_size, output_size):
    np.random.seed(0xc0ffee)
    W1 = np.random.randn(hidden_size, input_size) * 0.01
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size) * 0.01
    b2 = np.zeros((output_size, 1))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

def leaky_relu(x):
    neg = np.minimum(0,x) * 0.02
    pos = np.maximum(0,x)
    return neg + pos

def mse_loss(A2, Y):
    m = Y.shape[1]
    loss = (1 / (2 * m)) * np.sum(np.square(A2 - Y))
    return loss

def backward_propagation(X, Y, model, cache):
    m = X.shape[1]
    A1, A2 = cache["A1"], cache["A2"]
    W2 = model["W2"]
    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (np.sign(A1))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

def update_model(model, grads, learning_rate):
    W1, b1, W2, b2 = model["W1"], model["b1"], model["W2"], model["b2"]
    dW1, db1, dW2, db2 = grads["dW1"], grads["db1"], grads["dW2"], grads["db2"]
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

def train_neural_network(train_dataset, input_size, output_size,
                         hidden_size, epochs, learning_rate):
    model = initialize_model(input_size, hidden_size, output_size)
    loss = []
    count = len(train_dataset)
    for i in tqdm(range(epochs), desc="Training Progress"):
        cost = 0
        for xx in train_dataset:
            images = xx[:, :256]
            Y = xx[:, 256:].T
            X = images.reshape(images.shape[0], -1).T
            cache = forward_propagation(X, model)
            cost += mse_loss(cache["A2"], Y)
            grads = backward_propagation(X, Y, model, cache)
            model = update_model(model, grads, learning_rate)
        cost /= count
        loss.append(cost)
    return model, loss

epochs = 25
learning_rate = 0.01
input_size = 256
hidden_size = 64
output_size = 10
model, loss = train_neural_network(train_dataset, input_size, output_size,
hidden_size, epochs, learning_rate)
w1 = model['W1']
b1 = model['b1']
w2 = model['W2']
b2 = model['b2']
plt.figure(figsize=(6, 3))
plt.plot(loss)
plt.title("Loss")
plt.show()
print("Final Training Accu:",format(1-loss[-1], ".2%"))
print("Final Training Loss:", format(loss[-1], ".2%"))

def inference(model, X):
    W1, b1, W2, b2 = model["W1"], model["b1"], model["W2"], model["b2"]
    Z1 = np.dot(W1, X) + np.transpose(b1)
    A1 = np.transpose(leaky_relu(Z1))
    Z2 = np.dot(W2, A1) + b2
    A2 = leaky_relu(Z2)
    return A2

def predict_number(model, ip):
    o = inference(model, ip)
    num = np.argmax(o)
    return num

def validate_model(model, data):
    tsize = data.shape[0]
    images = data[:, :256]
    labels = argmax_labels(data[:, 256:])
    images = images.reshape(images.shape[0], -1)
    success = 0
    for i in tqdm(range(tsize), desc="Validating F128 Model\t"):
        out = predict_number(model, images[i, :])
        success += 1 if (out == labels[i]) else 0
    acc = success / tsize
    return acc

# Fixed-Point Representation Conversion as Used
def ndarray_to_fixed(arr: np.ndarray, bit_width: int, abs_error: float):
    def float_to_fixed(x, bit_width, min_accuracy):
        x = np.array(x, dtype=np.float64)
        frac_bits = int(np.ceil(-np.log2(min_accuracy)))
        while True:
            int_bits = bit_width - 1 - frac_bits
            if int_bits < 0:
                raise ValueError("Not enough bits to satisfy accuracy")
            max_val = (2**(int_bits)) - 2**(-frac_bits)
            min_val = -(2**(int_bits))
            if np.all((x >= min_val) & (x <= max_val)):
                break
            frac_bits -= 1
        scale = 2**frac_bits
        fixed_val = np.round(x * scale).astype(np.int64)
        return fixed_val, frac_bits
    if not isinstance(arr, np.ndarray):
        raise TypeError("Input must be a numpy.ndarray")
    best_frac_bits = None
    for val in arr.flatten():
        _, frac_bits = float_to_fixed(val, bit_width, abs_error)
        if best_frac_bits is None or frac_bits > best_frac_bits:
            best_frac_bits = frac_bits
    scale = 1 << best_frac_bits
    fixed_arr = np.round(arr * scale).astype(int)
    min_val = -(1 << (bit_width - 1))
    max_val = (1 << (bit_width - 1)) - 1
    fixed_arr = np.clip(fixed_arr, min_val, max_val)
    return fixed_arr, best_frac_bits

def heatmap_error(p, a_actual, a_fp, a_fb, err):
    temp = a_fp / (1 << a_fb)
    temp = a_actual - temp
    im = p.imshow(temp, cmap="plasma", vmin=(-1 * err), vmax=err)
    plt.colorbar(im)

bw = 8
err = 0.05
q_type = ""
wfp1, wfr1 = ndarray_to_fixed(w1, bw, err)
bfp1, bfr1 = ndarray_to_fixed(b1, bw, err)
wfp2, wfr2 = ndarray_to_fixed(w2, bw, err)
bfp2, bfr2 = ndarray_to_fixed(b2, bw, err)
model_fp = {"W1": wfp1, "WF1": wfr1, "b1": bfp1, "bf1": bfr1,
"W2": wfp2, "WF2": wfr2, "b2": bfp2, "bf2": bfr2}
if wfr1 == bfr1 == wfr2 == bfr2:
q_type = f"Q{bw-wfr1}.{wfr1}"
table = get_table(0)
display(md(eval(table)))

# Fixed-Point Hardware Model Arithmetic
def addfp(a, a_frac, b, b_frac, out_frac, a_width=8, b_width=8, out_width=9):
    def mask(x, bits):
        return x & ((1 << bits) - 1)
    def to_signed(x, bits):
        x = mask(x, bits)
        if x & (1 << (bits - 1)):
            return x - (1 << bits)
        return x
    a = to_signed(a, a_width)
    b = to_signed(b, b_width)
    if a_frac > b_frac:
        b <<= (a_frac - b_frac)
        frac = a_frac
    elif b_frac > a_frac:
        a <<= (b_frac - a_frac)
        frac = b_frac
    else:
        frac = a_frac
    res = a + b
    shift = frac - out_frac
    if shift >= 0:
        res >>= shift
    else:
        res <<= -shift
    res = to_signed(res, out_width)
    return res, out_frac

def mulfp(a, b, in_width=8, in_frac=5, out_width=16, out_frac=5):
    def mask(x, bits):
        return x & ((1 << bits) - 1)
    def to_signed(x, bits):
        x = mask(x, bits)
        if x & (1 << (bits - 1)):
            return x - (1 << bits)
        return x
    prod = a * b
    shift = (2 * in_frac) - out_frac
    if shift >= 0:
        prod >>= shift
    else:
        prod <<= -shift
    prod = to_signed(prod, out_width)
    return prod, out_frac

def macfp(W, X, in_frac, out_frac, in_width=8, out_width=16):
    acc = 0
    for wi, xi in zip(W, X):
        prod, prod_frac = mulfp(a=wi, b=xi, in_frac=in_frac, out_frac=out_frac, in_width=in_width, out_width=out_width)
        acc = acc + prod
    return acc, out_frac

def neuronfp(W, X, b, in_frac, out_frac, in_width=8, out_width=8):
    acc, acc_frac = macfp(W=W, X=X, in_frac=in_frac, out_frac=out_frac, in_width=in_width, out_width=16)
    out, out_frac = addfp(acc, acc_frac, b, in_frac, out_frac, a_width=16, b_width=in_width, out_width=out_width)
    leaky = out >> 3
    out = out if out >= leaky else leaky
    def mask(x, bits):
        return x & ((1 << bits) - 1)
    def to_signed(x, bits):
        x = mask(x, bits)
        if x & (1 << (bits - 1)):
            return x - (1 << bits)
        return x
    out = to_signed(out, out_width)
    return out.item()

def neuronfp_1h(W, X, b, in_frac, out_frac, in_width=8, out_width=8):
    acc = 0
    for wi, xi in zip(W, X):
        acc = acc + (wi if xi else 0)
    out, out_frac = addfp(acc, out_frac, b, in_frac, out_frac,
                          a_width=16, b_width=in_width,
                          out_width=out_width)
    leaky = out >> 3
    out = out if out >= leaky else leaky
    def mask(x, bits):
        return x & ((1 << bits) - 1)
    def to_signed(x, bits):
        x = mask(x, bits)
        if x & (1 << (bits - 1)):
            return x - (1 << bits)
        return x
    out = to_signed(out, out_width)
    return out.item()

def inference_fp(model, X, in_frac=5, bw = 8, out_frac=6):
    W1, W1_frac = model["W1"], model["WF1"]
    b1, b1_frac = model["b1"], model["bf1"]
    W2, W2_frac = model["W2"], model["WF2"]
    b2, b2_frac = model["b2"], model["bf2"]
    A1 = np.zeros(len(W1), dtype=np.int16)
    A2 = np.zeros(len(W2), dtype=np.int16)
    for i in range(len(W1)):
        A1[i] = neuronfp_1h(W1[i], X, b1[i], in_frac, in_frac, in_width=bw)
    for i in range(len(W2)):
        A2[i] = neuronfp(W2[i], A1, b2[i], in_frac, out_frac, in_width=bw)
    return A2

def predict_number_fp(model, ip, bw):
    o = inference_fp(model, ip, 5, bw=bw)
    num = np.argmax(o)
    return num

def validate_model_fp(model, data, datafr, bw):
    tsize = data.shape[0]
    images = data[:, :256]
    labels = argmax_labels(data[:, 256:])
    images = images.reshape(images.shape[0], -1)
    success = 0
    for i in tqdm(range(tsize), desc=f"Validating {q_type} Model"):
        out = predict_number_fp(model, images[i, :], bw)
        success += 1 if (out == labels[i]) else 0
    acc = success / tsize
    return acc


datafp, dfr = ndarray_to_fixed(test, bw, err)
fp128 = validate_model(model, test)
q3_5 = validate_model_fp(model_fp, datafp, dfr, bw)
table = get_table(1)
display(md(eval(table)))


index = 1
i = test[:,:256][index]
n = test[:,256:][index]
print("Actual Number:", np.argmax(n))
print("Predicted Number:", predict_number_fp(model_fp, i, bw))
plt.figure(figsize=(2,2))
plt.axis('off')
plt.imshow(i.reshape(16,16), aspect="auto", cmap="binary")
plt.show()
