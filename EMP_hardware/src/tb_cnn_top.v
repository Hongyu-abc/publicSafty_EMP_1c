`timescale 1ns/1ps

module tb_cnn_top;

    parameter NDIM  = 4; // 4x4 square
    parameter KDIM  = 3;
    parameter DW    = 8;
    parameter ACC_W = 20; // 9 INT8xINT8: 9*128*128 = 147456 

    localparam KLEN   = KDIM * KDIM;             // 9  -> k
    localparam ODIM   = NDIM - KDIM + 1;         // 
    localparam M      = ODIM * ODIM;             // 
    localparam N      = 1;
    localparam IN_AW = $clog2((NDIM*NDIM > KLEN*N) ? NDIM*NDIM : KLEN*N);
    localparam OUT_AW = (M*N > 1) ? $clog2(M*N) : 1;

    reg clk, rst_n;
    reg start;
    wire done;
    reg ld_en, ld_sel_ab; // ld_sel_ab = 1 -> kernel, 0 -> featureMap
    reg [IN_AW-1:0] ld_addr;
    reg signed [DW-1:0] ld_data;
    reg rd_en;
    reg [OUT_AW-1:0] rd_addr;
    wire signed [ACC_W-1:0] rd_data;

    cnn_top #(
        .M(M), .K(KLEN), .N(N), .W(NDIM), .H(NDIM), .DW(DW), .ACC_W(ACC_W)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .start(start), .done(done),
        .ld_en(ld_en), .ld_sel_ab(ld_sel_ab),
        .ld_addr(ld_addr), .ld_data(ld_data),
        .rd_en(rd_en), .rd_addr(rd_addr), .rd_data(rd_data)
    );

    always #5 clk = ~clk;

    reg signed [DW-1:0] ref_f [0:NDIM*NDIM-1]; // featureMap
    reg signed [DW-1:0] ref_k [0:KLEN-1];      // kernel
    reg signed [31:0]   ref_o [0:M-1];         // output

    integer h, w, oh, ow;                       // geo
    integer errors, checks;
    integer test_num, err_before;
    integer trial;                              // FIXED: Declared missing variable
    reg hung;

    // ------------------------------------------------------------------
    // Load tasks
    // ------------------------------------------------------------------
    task load_kernel;
        integer li;
        begin
            @(negedge clk);
            for (li = 0; li < KLEN; li = li + 1) begin
                ld_en     = 1'b1;
                ld_sel_ab = 1'b1;
                ld_addr   = li[IN_AW-1:0];
                ld_data   = ref_k[li];
                @(negedge clk);
            end
            ld_en = 1'b0;
        end
    endtask

    task load_fmap;
        integer li;
        begin
            @(negedge clk);
            for (li = 0; li < h*w; li = li + 1) begin
                ld_en     = 1'b1;
                ld_sel_ab = 1'b0;
                ld_addr   = li[IN_AW-1:0];
                ld_data   = ref_f[li];
                @(negedge clk);
            end
            ld_en = 1'b0;
        end
    endtask

    task compute_golden;
        integer oi, oj, ki, kj, acc;
        begin
            oh = h - KDIM + 1;
            ow = w - KDIM + 1;
            for (oi = 0; oi < oh; oi = oi + 1) begin
                for (oj = 0; oj < ow; oj = oj + 1) begin
                    acc = 0;
                    for (ki = 0; ki < KDIM; ki = ki + 1) begin
                        for (kj = 0; kj < KDIM; kj = kj + 1) begin
                            acc = acc + ref_f[(oi+ki)*w + (oj+kj)] * ref_k[ki*KDIM + kj];
                        end
                    end
                    ref_o[oi*ow + oj] = acc;
                end
            end
        end
    endtask

    task run_and_check;
        integer timeout;
        integer ri;
        input [8*40:1] name;    // limited to 40 characters
        begin
            test_num = test_num + 1;
            err_before = errors;
            compute_golden;

            load_kernel;
            load_fmap;

            @(negedge clk); start = 1'b1;
            @(negedge clk); start = 1'b0;

            // Check if DONE is asserted prematurely
            if (done !== 1'b0) begin
                $display("    ERROR: done asserted immediately after start");
                errors = errors + 1;
            end

            // Timeout wait for DONE
            timeout = 0;
            while (!done && timeout < 100000) begin
                @(posedge clk);
                timeout = timeout + 1;
            end
            if (timeout >= 100000) begin
                $display("[%0t] ERROR: done never asserted", $time);
                errors = errors + 1;
                hung = 1;
            end
            @(posedge clk);

            // Verify outputs
            for (ri = 0; ri < oh*ow; ri = ri + 1) begin
                rd_en   = 1'b1;
                rd_addr = ri[OUT_AW-1:0];
                #1;
                checks = checks + 1;
                if (rd_data !== ref_o[ri]) begin
                    errors = errors + 1;
                    if (errors - err_before <= 5) begin
                        $display("    MISMATCH out[%0d][%0d] (addr %0d): got=%0d expected=%0d",
                                 ri/ow, ri%ow, ri, rd_data, ref_o[ri]);
                    end
                end
            end
            rd_en = 1'b0;

            if (errors == err_before)
                $display("[PASS] test %0d: %0s  (%0dx%0d -> %0dx%0d)",
                         test_num, name, h, w, oh, ow);
            else
                $display("[FAIL] test %0d: %0s  (%0dx%0d -> %0dx%0d)  %0d mismatch(es)",
                         test_num, name, h, w, oh, ow, errors - err_before);
        end
    endtask

    // Helpers
    task set_kernel;    // row-major: k00 k01 k02 k10 k11 k12 k20 k21 k22
        input integer k0,k1,k2,k3,k4,k5,k6,k7,k8;
        begin
            ref_k[0]=k0; ref_k[1]=k1; ref_k[2]=k2;
            ref_k[3]=k3; ref_k[4]=k4; ref_k[5]=k5;
            ref_k[6]=k6; ref_k[7]=k7; ref_k[8]=k8;
        end
    endtask

    //1,2,3,4,...
    task fill_fmap_ramp;   
        integer r,c; begin
            for (r=0;r<h;r=r+1) 
                for (c=0;c<w;c=c+1) 
                    ref_f[r*w+c] = r*w+c+1; 
        end 
    endtask

    //[1,2,3][1,2,3][1,2,3]
    task fill_fmap_hramp;  
        integer r,c; begin      
            for (r=0;r<h;r=r+1) 
                for (c=0;c<w;c=c+1) 
                    ref_f[r*w+c] = c; 
        end 
    endtask

    //[1,1,1][2,2,2][3,3,3]
    task fill_fmap_vramp;  
        integer r,c; begin      
            for (r=0;r<h;r=r+1) 
                for (c=0;c<w;c=c+1) 
                    ref_f[r*w+c] = r; 
        end 
    endtask

    //v,v,v,v,v...
    task fill_fmap_const;  
        input integer v; 
        integer li; begin
            for (li=0;li<h*w;li=li+1) 
                ref_f[li] = v; 
        end 
    endtask

    task fill_fmap_rand;   
        integer li; begin
            for (li=0;li<h*w;li=li+1) 
                ref_f[li] = rand8(0); 
        end 
    endtask

    task fill_kernel_rand; 
        integer li; begin
            for (li=0;li<KLEN;li=li+1) 
                ref_k[li] = rand8(0); 
        end 
    endtask

    function signed [DW-1:0] rand8;
        input dummy;
        reg [31:0] r;
        begin
            r = $random;
            rand8 = r[7:0];
        end
    endfunction

    // Main Test Execution
    initial begin
        $dumpfile("waves.vcd");
        $dumpvars(0, tb_cnn_top);

        errors = 0;
        checks = 0;
        hung   = 0;
        clk = 0; rst_n = 0;
        start = 0; ld_en = 0; ld_sel_ab = 0; ld_addr = 0; ld_data = 0;
        rd_en = 0; rd_addr = 0;
        test_num = 0;

        h = NDIM;
        w = NDIM;

        repeat (3) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        // Directed Tests
        $display("---- Directed 1: impulse kernel ----");
        fill_fmap_ramp;  set_kernel(0,0,0, 0,1,0, 0,0,0);
        run_and_check("impulse kernel");

        $display("---- Directed 2: all-zeros kernel ----");
        fill_fmap_rand;  set_kernel(0,0,0, 0,0,0, 0,0,0);
        run_and_check("zero kernel");

        $display("---- Directed 3: all-ones kernel ----");
        fill_fmap_ramp;  set_kernel(1,1,1, 1,1,1, 1,1,1);
        run_and_check("box-sum");

        $display("---- Directed 4: sobel-asymmetric kernel ----");
        fill_fmap_hramp; set_kernel(1,0,-1, 2,0,-2, 1,0,-1);
        run_and_check("Sobel-X on horizontal ramp");
        fill_fmap_vramp; set_kernel(1,2,1, 0,0,0, -1,-2,-1);
        run_and_check("Sobel-Y on vertical ramp");

        $display("---- Directed 5: off-center impulse ----");
        fill_fmap_ramp; set_kernel(0,0,1, 0,0,0, 0,0,0);
        run_and_check("impulse at kernel[0][2]");
        fill_fmap_ramp; set_kernel(0,0,0, 0,0,0, 1,0,0);
        run_and_check("impulse at kernel[2][0]");

        $display("---- Directed 6: extreme values ----");
        fill_fmap_const(-128); set_kernel(-128,-128,-128,-128,-128,-128,-128,-128,-128);
        run_and_check("acc bound +147456");

        fill_fmap_const(-128); set_kernel(127,127,127,127,127,127,127,127,127);
        run_and_check("acc bound -146304");

        $display("---- Directed 7: back-to-back w/ no reset ----");
        fill_fmap_const(1); set_kernel(1,1,1,1,1,1,1,1,1);
        run_and_check("b2b A: all-ones (expect 9)");
        fill_fmap_const(0); set_kernel(1,1,1,1,1,1,1,1,1);
        run_and_check("b2b B: all-zero (catches stale output)");
        fill_fmap_rand; fill_kernel_rand;
        run_and_check("b2b C: random after zero");

        $display("---- Directed 8: mid-compute reset ----");
        fill_fmap_rand; fill_kernel_rand; compute_golden;
        load_kernel; load_fmap;
        @(negedge clk); start = 1'b1;
        @(negedge clk); start = 1'b0;
        repeat (10) @(posedge clk);
        rst_n = 0; repeat (3) @(posedge clk); rst_n = 1; @(posedge clk);
        fill_fmap_rand; fill_kernel_rand;
        run_and_check("recovery after mid-compute reset");

        // Randomized Trials
        $display("---- Randomized trials ----");
        for (trial = 0; trial < 30; trial = trial + 1) begin
            fill_fmap_rand; fill_kernel_rand;
            run_and_check("randomized");
        end

        // Final Results Summary
        $display("========================================");
        $display("Tests run: %0d    Value checks: %0d    Errors: %0d", test_num, checks, errors);
        if (hung)
            $display("RESULT: FAIL (design hung -- done never asserted)");
        else if (errors > 0) 
            $display("RESULT: FAIL");
        else                 
            $display("RESULT: PASS");
        $display("========================================");
        $finish;
    end

    // FIXED: Combined into a single Watchdog initial block
    initial begin
        #20000000;
        $display("========================================");
        $display("ERROR: TIMEOUT - simulation did not finish in time");
        $display("RESULT: FAIL (watchdog timeout -- simulation never completed)");
        $display("========================================");
        $finish;
    end

endmodule
