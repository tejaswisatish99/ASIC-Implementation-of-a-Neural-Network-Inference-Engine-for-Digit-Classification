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

##### --LEC Report
<img width="717" height="333" alt="image" src="https://github.com/user-attachments/assets/e2aa09d6-5507-4d7c-a01a-e9455a3d89aa" />

##### --Picture of Schematic from Genus (post-synthesis)
<img width="466" height="457" alt="image" src="https://github.com/user-attachments/assets/51ea1f25-0f17-4f51-9f0e-116d48b590c2" />


##### --QOR and Power Reports from Synthesis

<img width="313.5" height="479.5" alt="image" src="https://github.com/user-attachments/assets/e5847259-a1ec-4cb2-a5ef-85b7c615790a" />
<img width="521" height="150" alt="image" src="https://github.com/user-attachments/assets/739a13b7-b5d4-4e04-9ec7-4cb14d982585" />


##### --Placement
<img width="619" height="388.5" alt="image" src="https://github.com/user-attachments/assets/d8644dec-9f52-4728-ba4a-cd6e4c07d49f" />


##### --Routing
<img width="616" height="445" alt="image" src="https://github.com/user-attachments/assets/05165f2b-34c8-4300-9d6a-9fea7724d8f0" />


##### --Setup & Hold reports post route
<img width="420.5" height="235" alt="image" src="https://github.com/user-attachments/assets/e2e71c41-fd51-4212-bbf9-eaba83b308cf" />

##### --GDS Streamout
<img width="272" height="408" alt="image" src="https://github.com/user-attachments/assets/b8449dde-7285-4b71-852f-4c28cce258c2" />
