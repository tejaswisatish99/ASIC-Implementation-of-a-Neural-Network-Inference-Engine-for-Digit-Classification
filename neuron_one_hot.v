module neuronfp_1h #(parameter WIDTH = 8, N = 1, IFR = 5, OFR = 6, ACT = 2)
(W, X, b, out, done, ack, start, clk, rst);
input wire signed [N*WIDTH-1:0] W;
input wire [N-1:0] X;
input wire signed [WIDTH-1:0] b;
input wire clk, rst, start;
output wire signed [WIDTH-1:0] out;
output reg done, ack;
reg signed [N-1:0] x_local;
wire signed [2*WIDTH-1:0] sum;
wire signed [2*WIDTH-1:0] acc_next[0:N];
assign acc_next[0] = {2*WIDTH{1'b0}};
genvar i;
generate
for (i = 0; i < N; i = i + 1)
begin : mac_1h
wire signed [2*WIDTH-1:0] prod;
wire signed [2*WIDTH-1:0] w_ext =
{{WIDTH{W[i*WIDTH+WIDTH-1]}}, W[i*WIDTH +: WIDTH]};
assign prod = x_local[i] ? w_ext : {2*WIDTH{1'b0}};
add #(.WIDTH(2*WIDTH))
u_add (.a(acc_next[i]), .b(prod), .s(acc_next[i+1]));
end
endgenerate

addfp #(.AWIDTH(2*WIDTH), .A_FR(OFR), .BWIDTH(WIDTH),
.B_FR(IFR), .OWIDTH(2*WIDTH), .O_FR(OFR))
u_add (.a(acc_next[N]), .b(b), .s(sum));
generate
if (ACT == 1)
begin: ACT_RELU
assign out = (sum > 0) ? sum[WIDTH-1:0] : 0;
end
else if (ACT == 2)
begin: ACT_LEAKY_RELU
assign out = (sum > 0) ?
sum[WIDTH-1:0] : sum[WIDTH+2 -: WIDTH];
end
else
begin: ACT_LINEAR
assign out = sum[WIDTH-1:0];
end
endgenerate

always @(posedge clk or posedge rst)
begin
if(rst)
begin
ack <= 0;
x_local <= {N{1'b0}};
done <= 0;
end
else if(start)
begin
ack <= 1;
x_local <= X;
done <= 1;
end
else
begin
ack <= 0;
done <= 0;
end
end
endmodule

