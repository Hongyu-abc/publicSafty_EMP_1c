module mac_unit #(
    parameter DW = 8,
    parameter ACC_W = 32
)(
    input clk,
    input rst_n,

    input signed [DW-1:0] a,
    input signed [DW-1:0] b,

    input valid_in,
    input clear_acc,

    output signed [ACC_W-1:0] acc_out,
    output reg valid_out
);
    localparam HW = DW/2;   //half width
    // ====================================
    // Stage 1 Pipeline Registers
    // ====================================

    //2 half-width product
    //divide a 8*8 mac into 2 8*4 macs
    reg signed [(2*DW)-1:0] pp_hi;
    reg signed [(2*DW)-1:0] pp_lo;
    reg valid_1a, clear_1a;

    // Stores multiplication result
    reg signed [(2*DW)-1:0] product_reg;

    // Delayed control signals
    reg valid_reg;
    reg clear_reg;

    // ====================================
    // Stage 2 Accumulator
    // ====================================

    reg signed [ACC_W-1:0] accumulator;

    // ====================================
    // Sign extend product to accumulator width
    // ====================================

    wire signed [ACC_W-1:0] product_extended;
    assign product_extended =
        {{(ACC_W-(2*DW)){product_reg[(2*DW)-1]}}, product_reg};

    // ====================================
    // Two-stage MAC Pipeline
    // ====================================

    always @(posedge clk) begin
        if (!rst_n) begin
            // Reset Stage 1
            pp_hi <= 0;
            pp_lo <= 0;
            valid_1a <= 0;
            clear_1a <= 0;

            product_reg <= 0;
            valid_reg <= 0;
            clear_reg <= 0;

            // Reset Stage 2
            accumulator <= 0;
            valid_out <= 0;
        end else begin

            // ============================
            // STAGE 1
            // ============================

            //two 8*4 multiplications
            pp_hi <= $signed(a) * $signed(b[DW-1:HW]);        // signed
            pp_lo <= $signed(a) * $signed({1'b0, b[HW-1:0]}); // add a zero so that it becomes non-negative
            valid_1a <= valid_in;
            clear_1a <= clear_acc;

            // Carry metadata with product
            product_reg <= (pp_hi <<< HW) + pp_lo;
            valid_reg <= valid_1a;
            clear_reg <= clear_1a;

            // ============================
            // STAGE 2
            // ============================

            // Valid signal has same latency as data
            valid_out <= valid_reg;

            if (valid_reg) begin
                if (clear_reg) begin
                    // Start a new accumulation
                    accumulator <= product_extended;
                end else begin
                    // Add product to current sum
                    accumulator <= accumulator + product_extended;
                end
            end
        end
    end

    assign acc_out = accumulator;
endmodule
