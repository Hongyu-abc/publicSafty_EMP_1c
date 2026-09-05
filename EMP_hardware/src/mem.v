module mem #(
    parameter DW = 8,
    parameter DEPTH = 64
)(
    input clk,
    // write port
    input we,
    input [$clog2(DEPTH)-1:0] waddr,
    input signed [DW-1:0] wdata,
    // read port
    input [$clog2(DEPTH)-1:0] raddr,
    output signed [DW-1:0] rdata
);

    // actual storage
    reg signed [DW-1:0] memory [0:DEPTH-1];

    // Writing
    always @(posedge clk) begin
        if (we) begin
            memory[waddr] <= wdata;
        end
    end

    // Reading
    assign rdata = memory[raddr];

endmodule
