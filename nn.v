module nn (xi, yi, done, ack, start, clk , rst);
localparam WIDTH = 8;
localparam NLAYERS = 3;
localparam N_LIP = 256;
localparam N_LHI = 64;
localparam N_LOP = 10;
input wire start, clk, rst;
input wire [N_LIP-1:0] xi;
output wire [$clog2(N_LOP)-1:0] yi;
output reg ack;
output wire done;
wire signed [N_LHI*WIDTH-1:0] l1_out;
wire signed [N_LOP*WIDTH-1:0] l2_out;
wire [NLAYERS-1:0] l_done;
reg [NLAYERS-1:0] l_start;
wire [NLAYERS-1:0] l_ack;
assign done = l_done[NLAYERS-1];
layer1 #(.WIDTH(WIDTH), .NI(N_LIP), .NO(N_LHI), .OFR(5))
u_l1 (.xi(xi), .yi(l1_out), .done(l_done[0]), .ack(l_ack[0]),
.start(l_start[0]), .clk(clk), .rst(rst));
layer2 #(.WIDTH(WIDTH), .NI(N_LHI), .NO(N_LOP), .OFR(6))
u_l2 (.xi(l1_out), .yi(l2_out), .done(l_done[1]), .ack(l_ack[1]),
.start(l_start[1]), .clk(clk), .rst(rst));

argmax #(.WIDTH(WIDTH), .SIZE(N_LOP), .P(N_LOP))
u_argmax (.data_flat(l2_out), .max_index(yi), .done(l_done[2]), .ack(l_ack[2]),
.start(l_start[2]), .rst(rst), .clk(clk));
reg [1:0] state;
localparam IDLE=2'd0, L1=2'd1, L2=2'd2, PRED=2'd3;
always @(posedge clk or posedge rst)
begin
if (rst)
begin
state <= IDLE;
l_start <= {NLAYERS{1'b0}};
ack <= 0;
end
else
begin
ack <= (start & state == IDLE) ? 1 : (!start & state != IDLE) ? 0 : ack;
case (state)
IDLE: begin
if (start)
begin
l_start[0] <= 1;
state <= L1;
end
end
L1: begin
if (l_ack[0]) l_start[0] <= 0;
if (l_done[0])
begin
l_start[1] <= 1;
state <= L2;
end
end
L2: begin
if (l_ack[1]) l_start[1] <= 0;
if (l_done[1])
begin
l_start[2] <= 1;
state <= PRED;
end
end
PRED: begin
if(l_ack[2]) l_start[2] <= 0;
if (l_done[2]) state <= IDLE;
end
endcase
end
end
endmodule
