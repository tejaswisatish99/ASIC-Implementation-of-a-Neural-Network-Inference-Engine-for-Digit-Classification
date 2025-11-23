module mul #(parameter WIDTH = 8)(a, b, p, done, ack, start, clk, rst);
input wire signed [WIDTH-1:0] a, b;
input wire clk, rst, start;
output reg [2*WIDTH-1:0] p;
output reg ack, done;
reg busy;
always @(posedge clk or posedge rst)
begin
if (rst)
begin
p <= 0;
ack <= 0;
done <= 0;
busy <= 0;
end
else
begin
done <= 0;
if (start && !busy)
begin
p <= a * b;
ack <= 1;
busy <= 1;
end
else if (!start && busy)
begin
ack <= 0;
done <= 1;
busy <= 0;
end
end
end
endmodule