module macfp #(parameter WIDTH = 8, N = 2, IFR = 4, OFR = 5)(a, b, acc, done, ack, start, clk, rst);
input wire signed [N*WIDTH-1:0] a, b;
input wire clk, rst, start;
output wire signed [2*WIDTH-1:0] acc;
output wire done, ack;
wire signed [WIDTH-1:0] a_arr [0:N-1];
wire signed [WIDTH-1:0] b_arr [0:N-1];
genvar i;
generate
for (i = 0; i < N; i = i + 1) begin : unpack
assign a_arr[i] = a[i*WIDTH +: WIDTH];
assign b_arr[i] = b[i*WIDTH +: WIDTH];
end
endgenerate
wire signed [2*WIDTH-1:0] acc_next[0:N];
wire [N-1:0] mul_done;
wire [N-1:0] mul_ack;
assign acc_next[0] = {2*WIDTH{1'b0}};
assign acc = acc_next[N];
assign ack = &mul_ack;
assign done = &mul_done;
generate
for (i = 0; i < N; i = i + 1)
begin : mac
wire signed [2*WIDTH-1:0] prod;
mulfp #(.WIDTH(WIDTH), .OWIDTH(2*WIDTH), .IFR(IFR), .OFR(OFR))
u_mul (.a(a_arr[i]), .b(b_arr[i]), .p(prod), .done(mul_done[i]),
.ack(mul_ack[i]), .start(start), .clk(clk), .rst(rst));
add #(.WIDTH(2*WIDTH))
u_add(.a(acc_next[i]), .b(prod), .s(acc_next[i+1]));
end
endgenerate
endmodule