`ifndef NEURON
`define NEURON neuronfp
`endif
module `LAYER #(parameter WIDTH = 8, NI = 1, NO = 1, IFR = 5, OFR = 6, ACT = 2)
(xi, yi, done, ack, start, clk, rst);
`ifndef ONEHOTIP
input wire signed [NI*WIDTH-1:0] xi;
`else
input wire [NI-1:0] xi;
`endif
output signed [(NO*WIDTH)-1:0] yi;
input wire clk, start, rst;
output done, ack;
wire [WIDTH-1:0] fmem_w[0:NI*NO-1];
wire [NI*WIDTH-1:0] mem_w[0:NO-1];
wire [WIDTH-1:0] mem_b[0:NO-1];
wire signed [WIDTH-1:0] y_arr[0:NO-1];
wire [NO-1:0] n_done;
wire [NO-1:0] n_ack;
assign done = &n_done;
assign ack = &n_ack;
`include `LAYER_W_FILE
`include `LAYER_B_FILE

genvar n, j;
generate
for (n = 0; n < NO; n = n + 1)
begin
for (j = 0; j < NI; j = j + 1)
begin
assign mem_w[n][(j+1)*WIDTH-1 -: WIDTH] = fmem_w[n*NI + j];
end
end
endgenerate
genvar i;
generate
for (i = 0; i < NO; i = i + 1)
begin : unpack
assign yi[(i+1)*WIDTH-1 -: WIDTH] = y_arr[i];
end
endgenerate
generate
for (i = 0; i < NO; i = i + 1)
begin : neural_layer
`NEURON #(.WIDTH(WIDTH), .N(NI), .IFR(IFR),
.OFR(OFR), .ACT(ACT))
u_neuron (.W(mem_w[i]), .X(xi), .b(mem_b[i]),
.out(y_arr[i]), .done(n_done[i]), .ack(n_ack[i]),
.start(start), .clk(clk), .rst(rst));
end
endgenerate
endmodule