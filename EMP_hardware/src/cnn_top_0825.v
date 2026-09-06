
module cnn_top #(
    parameter M = (W-2)*(H-2),      // Output rows / windows 
    parameter K = 9,      // Inner dimension (3x3 kernel = 9)
    parameter N = 1,      // Output columns / filters, must be set to 1 only
    parameter W = 4,    //input width
    parameter H = 4,    //input depth
    parameter DW = 8,
    parameter ACC_W = 20
)(
    input clk,
    input rst_n,

    input start,
    output reg done,

    // Load interface
    input ld_en,
    input ld_sel_ab,

    // max(A,B) address width
    input [$clog2((M*K > K*N) ? M*K : K*N)-1:0] ld_addr,

    input signed [DW-1:0] ld_data,


    // Read output
    input rd_en,
    input [$clog2(M*N)-1:0] rd_addr,

    output signed [ACC_W-1:0] rd_data
);

// the address widths
localparam B_ADDR_W = $clog2(K*N);
localparam C_ADDR_W = $clog2(M*N);

//width & height of OUTPUT/fmap
localparam OW = W-2;
localparam OH = H-2;
localparam F_ADDR_W = $clog2(W*H);

// states for our FSM
localparam IDLE     = 2'd0;
localparam COMPUTE  = 2'd1;
localparam DRAIN    = 2'd2;
localparam FINISH   = 2'd3;

reg [1:0] state;

// our counters
reg [$clog2(M+1)-1:0] i;
reg [$clog2(N+1)-1:0] j;
reg [$clog2(K+1)-1:0] k;

// MAC
reg signed [DW-1:0] mac_a;
reg signed [DW-1:0] mac_b;

reg mac_valid;
reg mac_clear;

wire signed [ACC_W-1:0] mac_result;
wire mac_valid_out;

// pipeline tracking
reg [C_ADDR_W-1:0] addr_pipe0;
reg [C_ADDR_W-1:0] addr_pipe1;
reg [C_ADDR_W-1:0] addr_pipe2;
reg [C_ADDR_W-1:0] addr_pipe3;

reg final_pipe0;
reg final_pipe1;
reg final_pipe2;
reg final_pipe3;

// output counter
reg [$clog2(M*N+1)-1:0] outputs_done;

// memory signals
wire signed [DW-1:0] b_data;
wire signed [ACC_W-1:0] c_data;

// addresses
wire [B_ADDR_W-1:0] b_addr;
wire [C_ADDR_W-1:0] c_addr;

assign b_addr = (k*N)+j;
assign c_addr = (i*N)+j;

// memories
reg signed [DW-1:0] a_mem [0:W*H-1];
reg [F_ADDR_W:0] a_fill_ctr;
wire a_mem_full;

assign a_mem_full = (a_fill_ctr == W*H);

mem #(
    .DW(DW),
    .DEPTH(K*N)
)
B_MEM
(
    .clk(clk),

    .we(ld_en && ld_sel_ab),
    .waddr(ld_addr[B_ADDR_W-1:0]),
    .wdata(ld_data),

    .raddr(b_addr),
    .rdata(b_data)
);
mem #(
    .DW(ACC_W),
    .DEPTH(M*N)
)
C_MEM
(
    .clk(clk),

    .we(mac_valid_out && final_pipe3),
    .waddr(addr_pipe3),
    .wdata(mac_result),

    .raddr(rd_addr),
    .rdata(c_data)
);
// MAC instance
mac_unit #(
    .DW(DW),
    .ACC_W(ACC_W)
)
MAC
(
    .clk(clk),
    .rst_n(rst_n),

    .a(mac_a),
    .b(mac_b),

    .valid_in(mac_valid),
    .clear_acc(mac_clear),

    .acc_out(mac_result),
    .valid_out(mac_valid_out)
);

assign rd_data = c_data;

//addr pre-calculation
reg [F_ADDR_W-1:0]     win_base;   // fmap origin
reg [$clog2(OW+1)-1:0] oj;         // 0-OW-1, oj ? +1:+3
//---- next k ----
wire k_wrap = (k == K-1);
wire [$clog2(K+1)-1:0] next_k = k_wrap ? {$clog2(K+1){1'b0}} : k + 1'b1;

//---- next win_base ----
wire oj_wrap = (oj == OW-1);
wire [F_ADDR_W-1:0] next_win_base = !k_wrap ? win_base: oj_wrap ? win_base + 3: win_base + 1;

//---- next tap_off (next_k, not k）----
reg [F_ADDR_W-1:0] next_tap_off;
always @(*) case (next_k)
    0: next_tap_off = 0;
    1: next_tap_off = 1;
    2: next_tap_off = 2;
    3: next_tap_off = W;
    4: next_tap_off = W + 1;
    5: next_tap_off = W + 2;
    6: next_tap_off = 2*W;
    7: next_tap_off = 2*W + 1;
    default: next_tap_off = 2*W + 2;
endcase

//---- fmap addr: indexing -- a_mem -- on beat ----
reg [F_ADDR_W-1:0] fmap_addr_reg;

// the controller
always @(posedge clk or negedge rst_n)
begin
    if(!rst_n)
    begin
        state <= IDLE;
        done <= 0;
        i <= 0;
        j <= 0;
        k <= 0;
        outputs_done <= 0;
        win_base <= 0;
        oj <= 0;
        mac_a <= 0;
        mac_b <= 0;
        mac_valid <= 0;
        mac_clear <= 0;
        a_fill_ctr <= 0;
        fmap_addr_reg <= 0;
    end else begin
        if (ld_en && !ld_sel_ab) begin
            a_mem[ld_addr[F_ADDR_W-1:0]] <= ld_data;
            if (!a_mem_full)
                a_fill_ctr <= a_fill_ctr + 1'b1;
        end
        done <= 0;

        case(state)
            IDLE: begin
                mac_valid <= 0;
                mac_clear <= 0;
                win_base <= 0;
                oj <= 0;
                fmap_addr_reg <= 0;
                if (start && a_mem_full) begin
                    state <= COMPUTE;
                    outputs_done <= 0;
                end
            end

            COMPUTE: begin
                mac_a <= a_mem [fmap_addr_reg]; //addr pre-calc
                fmap_addr_reg <= next_win_base + next_tap_off;
                mac_b <= b_data;

                mac_valid <= 1;

                // first multiply of each dot product
                mac_clear <= (k == 0);

                // pipeline output address
                addr_pipe3 <= addr_pipe2;
                addr_pipe2 <= addr_pipe1;
                addr_pipe1 <= addr_pipe0;
                addr_pipe0 <= c_addr;

                final_pipe3 <= final_pipe2;
                final_pipe2 <= final_pipe1;
                final_pipe1 <= final_pipe0;
                final_pipe0 <= (k == K-1);

                if(mac_valid_out && final_pipe3)
                    outputs_done <= outputs_done + 1;

                // k loop
                if(k == K-1) begin
                    k <= 0;
                    // j loop
                    if(j == N-1) begin  //fmap movement
                       j <= 0;

                       // i loop
                       if(i == M-1) begin
                            state <= DRAIN;
                       end else begin
                            i <= i + 1;
                        end
                    end else begin
                        j <= j + 1;
                    end
                    //origin of fmap movement
                    if (oj == OW-1) begin
                        oj <= 0;
                        win_base <= win_base + 3;
                    end else begin
                        oj <= oj + 1;
                        win_base <= win_base + 1;
                    end
                end else begin
                    k <= k + 1;
                end
            end

            DRAIN: begin
                mac_valid <= 0;
                mac_clear <= 0;

                addr_pipe3 <= addr_pipe2;
                addr_pipe2 <= addr_pipe1;
                addr_pipe1 <= addr_pipe0;
                addr_pipe0 <= 0;

                final_pipe3 <= final_pipe2;
                final_pipe2 <= final_pipe1;
                final_pipe1 <= final_pipe0;
                final_pipe0 <= 0;

                if(mac_valid_out && final_pipe3) begin
                    outputs_done <= outputs_done + 1;
                end

                if(outputs_done == M*N-1 &&
                    mac_valid_out &&
                    final_pipe3) begin
                    state <= FINISH;
                end
            end

            FINISH: begin
                done <= 1;

                i <= 0;
                j <= 0;
                k <= 0;

                outputs_done <= 0;
                win_base <= 0;
                oj <= 0;
                fmap_addr_reg <= 0;

                state <= IDLE;

                a_fill_ctr <= 0;

                addr_pipe0 <= 0; addr_pipe1 <= 0; addr_pipe2 <= 0; addr_pipe3 <= 0;
                final_pipe0 <= 0; final_pipe1 <= 0; final_pipe2 <= 0; final_pipe3 <= 0;
            end
        endcase
    end
end
endmodule
