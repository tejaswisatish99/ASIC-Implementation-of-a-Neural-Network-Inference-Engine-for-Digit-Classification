`define LAYER layer1
`define LAYER_W_FILE "l1_w.vh"
`define LAYER_B_FILE "l1_b.vh"
`define ONEHOTIP 1
`define NEURON neuronfp_1h
`include "layer.vh"
`undef NEURON
`undef ONEHOTIP
`undef LAYER_W_FILE
`undef LAYER_B_FILE
`undef LAYER
`define LAYER layer2
`define LAYER_W_FILE "l2_w.vh"
`define LAYER_B_FILE "l2_b.vh"
`include "layer.vh"
`undef LAYER_W_FILE
`undef LAYER_B_FILE
`undef LAYER