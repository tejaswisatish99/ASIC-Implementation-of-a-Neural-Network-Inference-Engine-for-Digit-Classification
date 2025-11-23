# ASIC-Implementation-of-a-Neural-Network-Inference-Engine-for-Digit-Classification
Handwritten digit recognition with a Python-trained feedforward neural network (256-64-10), INT8 quantized inference in SystemVerilog, and full ASIC implementation with Cadence tools. Includes testbenches, synthesis flow, reports, and GDSII output.

With the semeion_top.m script, ran to train the neural network and got the floating point weights (w1, w2) and biases (b2, b3).
These floating-point parameters were quantized to get fixed point parameters and analyzed for precision loss by comparing the inference accuracy of the quantized model against the original floating-point model.
Trail and error method to arrive at the fractional part, started to fit total bit width as 8bits checked the accuracy to meet expected numbers.
Total memory needed for weights and biases (in bits).
##### w1 – 256x15 nodes with 8bit = 30720 bits
##### w2 – 15x10 nodes with 8bit = 1200 bits
##### b1 – 15nodes with 8bit = 120 bits
##### b2 – 10nodes with 8bit = 80 bits
##### Size of multiplier needed : 8*8 resulting into 16bit output

A fixed-point MAC (Multiply–Accumulate) unit with widened accumulators to balance precision, minimizing overflow and resource utilization while maintaining inference accuracy.

Implemented the nonlinear sigmoid function using a precomputed lookup table (ROM-based), eliminating the need for expensive exponential computations and significantly speeding up activation processing.

A dedicated controller for data flow, layer sequencing, and synchronization between MAC, memory, and output modules, ensuring deterministic timing and minimal control overhead.

##### --Picture of Schematic from Genus (post-synthesis)
<img width="932" height="914" alt="image" src="https://github.com/user-attachments/assets/51ea1f25-0f17-4f51-9f0e-116d48b590c2" />



