module addfp #(parameter AWIDTH = 5, A_FR = 4, BWIDTH = 5, B_FR = 4, OWIDTH = 6, O_FR = 4)(a, b, s);
input signed [AWIDTH-1:0] a;
input signed [BWIDTH-1:0] b;
output signed [OWIDTH-1:0] s;
localparam integer MAX_FR = (A_FR > B_FR) ? A_FR : B_FR;
localparam integer SHIFT = MAX_FR - O_FR;
function automatic signed [OWIDTH-1:0] signext_a;
input signed [AWIDTH-1:0] in;
begin
if(OWIDTH > AWIDTH)
signext_a = {{(OWIDTH-AWIDTH){in[AWIDTH-1]}}, in};
else if (OWIDTH == AWIDTH)
signext_a = in;
else
signext_a = in[OWIDTH-1:0];
end
endfunction
function automatic signed [OWIDTH-1:0] signext_b;
input signed [BWIDTH-1:0] in;
begin
if(OWIDTH > BWIDTH)
signext_b = {{(OWIDTH-BWIDTH){in[BWIDTH-1]}}, in};
else if (OWIDTH == BWIDTH)
signext_b = in;
else
signext_b = in[OWIDTH-1:0];
end
endfunction
wire signed [OWIDTH-1:0] a_ext = signext_a(a);
wire signed [OWIDTH-1:0] b_ext = signext_b(b);
wire signed [OWIDTH-1:0] a_align =
(A_FR < MAX_FR) ? (a_ext <<< (MAX_FR - A_FR)) : a_ext;
wire signed [OWIDTH-1:0] b_align =
(B_FR < MAX_FR) ? (b_ext <<< (MAX_FR - B_FR)) : b_ext;
wire signed [OWIDTH:0] sum = a_align + b_align;
assign s = (SHIFT >= 0) ? sum >>> SHIFT : sum <<< (-SHIFT);
endmodule
module mulfp #(parameter WIDTH = 8, IFR = 4, OWIDTH = 16, OFR = 5)(a, b, p, done, ack, start, clk, rst);
input wire signed [WIDTH-1:0] a, b;
input wire clk, rst, start;
output wire signed [OWIDTH-1:0] p;
output wire done, ack;
wire signed [(2*WIDTH)-1:0] prod;
localparam integer SHIFT = (2*IFR) - OFR;
assign p = (SHIFT >= 0) ? prod >>> SHIFT : prod <<< (-SHIFT);
mul #(.WIDTH(WIDTH))
mul_inst (.a(a), .b(b), .p(prod), .done(done),
.ack(ack), .start(start), .clk(clk), .rst(rst));
endmodule