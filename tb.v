`ifndef OUTPATH
`define OUTPATH .
`endif
`define OUTFILE `"`OUTPATH/wave.vcd`"
module tb;
localparam WIDTH = 8;
localparam N = 319;
localparam NI = 256;
integer wave;
integer sample;
integer index;
integer counter;
integer iter;
integer debug;
time t_start;
reg clk, rst, start;
reg [NI-1:0] mem_t [0:N-1];
wire [$clog2(10)-1:0] out;
wire done, ack;
reg [(WIDTH/2)-1:0] mem_o [0:N-1];
nn u_nn(.xi(mem_t[index]), .yi(out), .done(done),
.ack(ack), .start(start), .clk(clk), .rst(rst));
initial clk = 1;
always #1 clk = !clk;

initial
begin
if (!$value$plusargs("sample=%d", sample))
sample = -1;
if (!$value$plusargs("wave=%d", wave))
wave = 0;
if (!$value$plusargs("debug=%d", debug))
debug = 0;
if ((wave == 1 || debug == 1)&& sample != -1)
begin
$dumpfile(`OUTFILE);
$dumpvars(0, tb);
end
if (debug)
begin
$display("< ! > Loading Test Samples...");
$fflush();
end
$readmemb("../model/test/test_ip.mem", mem_t);
$readmemh("../model/test/test_op.mem", mem_o);
if (debug)
begin
$display("< / > Done!");
$fflush();
end
rst = 1; start = 0;
#1 rst = 0;

if (sample == -1)
begin
index = 0;
counter = 0;
$display("< ! > NN DUT Validation in progress...");
if (!debug) $write("< i > Progress: ");
$fflush();
t_start = $time;
for (iter = 0; iter < N; iter = iter + 1)
begin
index = iter;
start = 1; #1;
wait(ack);
start =0;
wait(done);
counter = (out == mem_o[iter]) ? counter + 1 : counter;
if (debug)
begin
$display("< %s > Test Sample[%3d/%3d]: A - %1d, P - %1d",
(out == mem_o[iter]) ? "/" : "x" , (iter+1), N, mem_o[iter], out);
$fflush();
end
else if ((iter % 15) == 0)
begin
$write("=");
$fflush();
end
end
if (!debug) $write("\n");
$display("< / > Done! Time Elapsed: %5t", $time - t_start);
$display("< i > Validation Accuracy: %.2f%%",
($itor(counter) / $itor(N))*100);
$fflush();
end
else
begin


if (debug)
begin
$display("< ! > Running... ");
$fflush();
end
index = sample;
t_start = $time;
start = 1; #0.5;
wait(ack);
start = 0;
wait(done);
if (debug)
begin
$display("< / > Done!");
$display("< ! > Test Sample[%3d/%3d]: A - %1d, P - %1d",
sample+1, N, mem_o[sample], out);
$fflush();
end
if(out == mem_o[sample])
$display("< / > Test Passed! Time Elapsed: %2t", $time - t_start);
else
$display("< x > Test Failed! Time Elapsed: %2t", $time - t_start);
$fflush();
end

#1 $finish(0);
end
endmodule

