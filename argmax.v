module argmax #(parameter WIDTH = 16, SIZE = 10, P = 4)
(data_flat, max_index, done, ack, start, clk, rst);
input wire rst, start, clk;
input wire signed [SIZE*WIDTH-1:0] data_flat;
output reg done, ack;
output reg [$clog2((SIZE>0)?SIZE:1)-1:0] max_index;
localparam IDLE = 1'b0;
localparam SCAN = 1'b1;
localparam signed [WIDTH-1:0] MIN_VAL = -(1 <<< (WIDTH-1));
reg state;
reg [$clog2(SIZE+P-1)-1:0] idx;
reg signed [WIDTH-1:0] max_val;
reg [$clog2(SIZE+P-1)-1:0] max_idx;
wire signed [WIDTH-1:0] vals [0:P-1];
genvar k;
generate
for (k = 0; k < P; k = k + 1)
begin : gen_vals
assign vals[k] =
((idx+k) < SIZE) ?
data_flat[(idx+k)*WIDTH +: WIDTH] : MIN_VAL;
end
endgenerate
reg signed [WIDTH-1:0] batch_max_val;
reg [$clog2(SIZE+P-1)-1:0] batch_max_idx;

integer i;
// verilator lint_off BLKSEQ
always @(max_val, max_idx, idx)
begin
batch_max_val = max_val;
batch_max_idx = max_idx;
for (i = 0; i < P; i = i + 1)
begin
if ((idx+$unsigned(i)) < SIZE && vals[i] > batch_max_val)
begin
batch_max_val = vals[i];
batch_max_idx = idx + $unsigned(i);
end
end
end
// verilator lint_on BLKSEQ
always @(posedge clk or posedge rst)
begin
if(rst)
begin
idx <= 0;
max_val <= MIN_VAL;
max_idx <= 0;
max_index <= 0;
done <= 0;
ack <= 0;
state <= IDLE;
end
else
begin
case (state)
IDLE: begin
ack <= 0;
max_val <= MIN_VAL;

done <= 0;
if (start)
begin
ack <= 1;
idx <= 0;
max_val <= data_flat[0 +: WIDTH];
max_idx <= 0;
state <= SCAN;
end
end
SCAN: begin
if (!start)
ack <= 0;
max_val <= batch_max_val;
max_idx <= batch_max_idx;
idx <= idx + P;
if (idx >= SIZE)
begin
state <= IDLE;
done <= 1;
max_index <= max_idx;
end
end
endcase
end
end
endmodule