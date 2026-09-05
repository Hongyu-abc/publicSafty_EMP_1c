# CNN Convolution Accelerator — Design and Optimization Report

**Design**: `cnn_top` — im2col-based 3×3 convolution accelerator
**Technology**: SkyWater sky130A / `sky130_fd_sc_hd`
**Flow**: OpenLane 2.3.10, Classic flow, RTL-to-GDS
**Final version**: `src/cnn_top_0825.v` @ 6.85 ns

---

## 0. Summary

| | Starting point | **Final** | |
|---|---|---|---|
| Fmax | 76.9 MHz | **146.0 MHz** | **+90%** |
| Die area | 144,584 µm² | **63,095 µm²** | **−56.4%** |
| Standard cells | 10,278 | **3,540** | −65.6% |
| Flip-flops | 672 | **406** | −39.6% |
| Power | 6.56 mW | 6.74 mW | +2.7% |
| Time per convolution (4×4) | 624 ns | **288 ns** | **2.17×** |
| Energy per convolution | 4.09 nJ | **1.94 nJ** | **−52.6%** |
| Hard signoff violations | 18 slew + 1 antenna | **all zero** | ✓ |

The optimization consisted of **4 RTL architectural changes** and **4 physical-design tuning steps**, each driven by post-PnR STA data.
A separate **5-way parallel MAC variant** (7×7 input) was also implemented, achieving a further 2.7× reduction in time per output element (Section 8).

---

## 1. Design Overview

### 1.1 Function

Computes the **cross-correlation** (not flipped convolution) of a `W×H` 8-bit signed feature map with a 3×3 8-bit signed kernel:

```
out[oi][oj] = Σ(ki=0..2) Σ(kj=0..2)  fmap[oi+ki][oj+kj] × kern[ki][kj]
```

Output dimensions are `OW×OH = (W−2)×(H−2)`; accumulated results are 20-bit signed.

### 1.2 Parameters

| Parameter | Value | Notes |
|---|---|---|
| `W` / `H` | 4 / 4 | Input dimensions, **compile-time parameters** |
| `M` | `(W−2)(H−2)` = 4 | Number of output elements (windows), derived automatically |
| `K` | 9 | Flattened kernel length (3×3) |
| `N` | 1 | Number of filters (fixed at 1 in this design) |
| `DW` | 8 | Data width (INT8) |
| `ACC_W` | 20 | Accumulator width |

**Derivation of `ACC_W = 20`**: nine INT8×INT8 multiply-accumulate terms have a maximum magnitude of `9 × 128 × 128 = 147,456`, which requires 19 signed bits (±262,143); 20 bits provides one bit of margin. This corresponds to technique 11 (bit-width reduction) in the course optimization guide, and directly eliminated 12 physical output pins (`rd_data`) relative to the original 32-bit accumulator.

### 1.3 Interface Contract

```
ld_sel_ab = 0 -> feature map,  ld_addr = row·W + col,   range 0 .. W·H−1
ld_sel_ab = 1 -> kernel,       ld_addr = ki·3 + kj,     range 0 .. 8
rd_addr        -> output,      = oi·(W−2) + oj,         range 0 .. M−1
rd_data is combinationally readable; done is a one-cycle pulse; no runtime size ports
```

### 1.4 Datapath

```
ld_data ──┬─(sel=0)─→ a_mem [W·H]  ──┐
          └─(sel=1)─→ B_MEM [9]     │
                          │          │
              b_addr = k  │          │  fmap_addr_reg = row_base + tap_off
                          ↓          ↓
                     mac_b ←──   mac_a
                          └──┬──┘
                             ↓
                    mac_unit (3-stage pipeline)
                    1a: pp_hi / pp_lo   (two 8×4 partial products)
                    1b: product_reg     (shift and combine)
                    2 : accumulator     (accumulate)
                             ↓
                        c_mem [M]  ──→ rd_data
```

**Where the matrix multiply lives**: `mac_unit` is the computational core of an `M×9` by `9×1` matrix multiplication. `a_mem` supplies the rows of matrix A (the nine pixels of each window) through implicit im2col addressing, `B_MEM` supplies the B vector (the flattened kernel), and `c_mem` collects the C vector (the dot-product result for each output element). The control FSM walks the entire multiplication in `M×K` cycles.

### 1.5 Implicit im2col Addressing

Rather than materializing the im2col matrix, the address is decomposed into three terms:

```
addr = (oi + ki)·W + (oj + kj)
     = [oi·W + oj]  +  [ki·W + kj]
       └ win_base ─┘   └ tap_off ┘   (nine compile-time constants)
```

`tap_off` is selected from a nine-entry constant table indexed by `k` (`0, 1, 2, W, W+1, W+2, 2W, 2W+1, 2W+2`); `win_base` advances by `+1` within a row of windows and by `+3` at a row boundary (the constant is independent of `W` because `OW = W−2` always holds).

---

## 2. Functional Verification

Self-checking testbench: `src/tb_cnn_top.v`

### 2.1 Structure

- **Golden model**: a naive four-level nested loop computing direct cross-correlation. It **deliberately does not use im2col** — if the reference model used the same reformulation as the RTL, both would share any indexing bug and the comparison would be worthless.
- The golden output array is declared **32 bits wide** (wider than `ACC_W = 20`), so an accumulator overflow produces a FAIL rather than truncating identically to the DUT and passing silently.
- Every test reports PASS/FAIL individually. The DUT is **not reset between tests**, so consecutive runs naturally expose leftover state.

### 2.2 Test List (43 tests, 172 value checks at 4×4)

| # | Test | Defect targeted |
|---|---|---|
| 1 | Centre impulse kernel | Almost all addressing / stride errors |
| 2 | Zero kernel | Accumulator not cleared between output elements |
| 3 | All-ones kernel (box sum) | Easy to verify by hand |
| 4–5 | Sobel-X / Sobel-Y | Kernel flipped or transposed |
| 6–7 | **Off-centre impulse ×2** (mutual transposes) | Kernel flattening order reversed — a centre impulse is invariant under transpose and cannot detect this |
| 8–9 | Accumulator bounds ±147,456 / −146,304 | Overflow from insufficient `ACC_W` |
| 10–12 | Back-to-back triple (all-ones → all-zero → random) | Stale output memory, counters not reset |
| 13 | Reset injected mid-compute | Reset recovery path |
| 14–43 | Randomized data ×30 | Unanticipated combinations |

### 2.3 Results

| Size | P (parallelism) | Result |
|---|---|---|
| 3×3 | 1 | PASS |
| 4×4 | 2 | PASS (43 tests / 172 checks / 0 errors) |
| 5×5 | 3 | PASS |
| 6×6 | 4 | PASS |
| 7×7 | 5 | PASS |

**Verification methodology**: before each RTL change, a known bug was deliberately injected into the golden model (e.g. changing `ki*KDIM+kj` to `kj*KDIM+ki`) to confirm the testbench actually FAILs, then reverted. This validates the testbench itself — the practice caught a defect in which the `errors` counter was never incremented, causing every test to report PASS unconditionally.

---

## 3. Final Implementation Results

### 3.1 Signoff: All Checks Clean

| Check | Result |
|---|---|
| Setup violations / TNS | **0** / 0.000 (all nine PVT corners positive) |
| Hold violations | **0** |
| Max slew (vs library 1.5 ns limit) | **0** |
| Max capacitance | **0** |
| Antenna (nets / pins) | **0 / 0** |
| Magic DRC / KLayout DRC / Routing DRC | **0 / 0 / 0** |
| LVS / XOR | **0 / 0** |
| Flow errors | **0** |

Setup slack per corner:

| Corner | Slack | | Corner | Slack |
|---|---|---|---|---|
| max_ss_100C_1v60 | **+0.042** ← worst | | max_tt_025C_1v80 | +2.667 |
| nom_ss_100C_1v60 | +0.117 | | nom_tt_025C_1v80 | +2.707 |
| min_ss_100C_1v60 | +0.207 | | min_tt_025C_1v80 | +2.733 |
| max_ff_n40C_1v95 | +3.130 | | nom/min_ff | +3.166 / +3.186 |

### 3.2 PPA

| Metric | Value |
|---|---|
| **Clock period / Fmax** | **6.85 ns / 146.0 MHz** |
| Setup / hold slack | +0.0418 / +0.1089 ns |
| Clock skew | 0.287 ns |
| **Die** | **245.9 × 256.6 µm = 63,095 µm²** (0.063 mm²) |
| Core area | 54,878 µm² |
| Cell area | 27,830 µm² |
| Utilization | 50.7% |
| **Power** | **6.736 mW** |
| ├ internal | 4.957 mW (73.6%) |
| ├ switching | 1.779 mW (26.4%) |
| └ leakage | 34 nW (0.0005%) |
| I/O pins | 45 |
| Wirelength / vias | 60,582 µm / 16,810 |
| IR drop (worst) | 0.91 mV |
| **Time per 4×4 convolution** | ~42 cycles = **288 ns** |
| **Energy per convolution** | **1.94 nJ** |

### 3.3 Cell Composition (3,540 standard cells)

| Class | Count | Share |
|---|---|---|
| Multi-input combinational | 1,252 | 35.4% |
| **Antenna diodes** | **560** | **15.8%** |
| Flip-flops | 406 | 11.5% |
| Timing repair buffers | 405 | 11.4% |
| Clock buffers | 59 | 1.7% |
| Other (inverters, simple gates) | 858 | 24.2% |
| *Fill cells 3,920 / tap cells 792 (physical fill, counted separately)* | | |

**Observation**: even after antenna threshold tuning, **15.8% of standard cells are diodes that implement no logic function**. This is the direct cost of passing antenna checks.

---

## 4. RTL Architectural Optimization

Four changes, each fully re-verified against all 43 tests across all five input sizes.

| Version | Change | ns | MHz | FF | Cells | Die µm² | mW | slew (lib) | ant |
|---|---|---|---|---|---|---|---|---|---|
| **V2** | `a_idx` counter replaces `i*K+k` | 13.0 | 76.9 | 672 | 10,278 | 144,584 | 6.56 | 18 | 1 |
| **V3** | Implicit im2col (remove `rearranged`) | 11.0 | 90.9 | 372 | 4,296 | 74,716 | 4.16 | **0** | **0** |
| **V4** | Address precomputation (retiming) | 10.0 | 100.0 | 376 | 4,408 | 75,811 | 4.73 | 0 | 0 |
| **V5** | MAC split into three pipeline stages | 8.25 | 121.2 | 406 | 4,503 | 77,186 | 5.83 | 0 | 0 |
| **V6** | `c_mem` write-enable gating | 8.25 | 121.2 | 406 | 4,549 | 77,470 | 5.72 | 0 | 0 |

### 4.1 V2: Replace Address Arithmetic with an Incrementing Counter

**Evidence**: post-PnR STA identified the critical path as `k counter → (a_addr = i·K + k multiply-add) → rearranged 36:1 read mux → mac_a`.

**Observation**: with `N = 1`, `j` is always 0, so each cycle advances only `k` (wrapping to 0 and incrementing `i` at `k = 8`). The actual sequence of `a_addr = i·9 + k` is therefore `0, 1, 2, …, 35` — **strictly sequential**. The multiply-add was computing a value that a simple up-counter provides for free.

**Change**: replaced `assign a_addr = (i*K)+k` with an `A_ADDR_W`-bit counter.

**Result**: critical path improved by **2.50 ns** (slack +1.248 → +3.751 at 13 ns) at a cost of six flip-flops. Cell area actually *decreased* by 0.1% — the eliminated adder was larger than the added register. Pipeline latency is unchanged, so `addr_pipe` required no adjustment.

### 4.2 V3: Implicit im2col (the single highest-impact change)

**Evidence**: the `rearranged` array (the materialized im2col matrix) occupied 288 flip-flops — **43% of all sequential cells** — and its contents were entirely duplicates of data already in `a_mem` (2.25× redundancy at 4×4, **4.6×** at 7×7). Its nine-port parallel write decoder was also the #2 critical path at the time.

**Change**: instead of materializing the im2col matrix, index `a_mem` directly using `addr = win_base + tap_off` (see §1.5). Removed `rearranged`, `offx`, `offy`, `count` and the entire `IM2COL` state.

**Result**:

| | Before | After |
|---|---|---|
| Die area | 144,584 µm² | **74,716 µm² (−48.3%)** |
| Flip-flops | 672 | **372 (−44.6%)** |
| Standard cells | 10,278 | 4,296 (−58.2%) |
| Power | 6.56 mW | **4.16 mW (−36.6%)** |
| Wirelength | 176,739 | 56,051 (−68.3%) |
| slew (library) / antenna | 18 / 1 | **0 / 0** |
| IM2COL phase cycles | 7 (4×4) / 31 (7×7) | **0** |

This is a change that improves **all three PPA axes simultaneously** — possible only because what was eliminated was pure data redundancy, not a trade-off.

### 4.3 V4: Address Precomputation (Retiming)

**Evidence**: after V3, the path `k counter → tap_off 9-way mux → adder → a_mem 16:1 read mux → mac_a` became #2 at 9.30 ns, trailing #1 (the multiplier at 9.56 ns) by only 0.26 ns.

**Change**: move the address arithmetic to the other side of a register boundary — compute the address required by cycle *n* combinationally during cycle *n−1* and register it, so the read path degenerates to "register → mux → register" with no arithmetic. **This is retiming (technique 10), not pipelining: latency is unchanged and `addr_pipe` depth required no adjustment.**

**Result (recorded honestly)**: viewed in isolation, this change was a **net loss**:

| | Before | After |
|---|---|---|
| Target path (address → mac_a) | 9.30 ns (#2) | **5.38 ns (#5)** ✓ |
| #1 critical path (multiplier) | 9.51 ns | 9.80 ns (**+0.27 ns**) ✗ |
| Standard cells | 4,311 | 4,408 (+97) ✗ |

Cause: the combinational logic for `next_k` / `next_win_base` was not shared with the FSM by the synthesizer, adding roughly 93 combinational cells. This grew the die by 1.5%, which lengthened wires and slowed even the untouched multiplier path.

**But it was the prerequisite for the next step**: it widened the gap between #1 and #2 from **0.26 ns to 2.74 ns**, changing the expected payoff of pipelining the multiplier from "at most +7%" to "up to +35%".

> **Methodological note**: evaluating a change in isolation can produce the wrong conclusion. Had this been reverted under a "no benefit, roll it back" policy, the subsequent 100 → 146 MHz improvement would have been unreachable.

### 4.4 V5: Split the MAC into Three Pipeline Stages

**Evidence**: after V4, #1 was the multiplier (9.80 ns) and #2 the accumulator (7.06 ns), a gap of 2.74 ns — only at this point does splitting the multiplier actually pay.

**Change**: using the signed decomposition `b = signed(b[7:4])·16 + b[3:0]`, split the single 8×8 multiply into two 8×4 partial products:

```verilog
// Stage 1a
pp_hi <= $signed(a) * $signed(b[DW-1:HW]);
pp_lo <= $signed(a) * $signed({1'b0, b[HW-1:0]});
// Stage 1b
product_reg <= (pp_hi <<< HW) + pp_lo;
// Stage 2 (unchanged)
accumulator <= clear_reg ? product_extended : accumulator + product_extended;
```

`addr_pipe` was correspondingly deepened from three to four stages (MAC latency 2 → 3 cycles).

**Result**: critical path **9.80 → 7.83 ns (−1.97 ns)**; setup slack at 10 ns jumped from +0.198 to +2.171. Latency of a single MAC increased by one cycle, but the COMPUTE phase still admits a new multiply term every cycle, so total cycle count increased by only one cycle in DRAIN (+2.4%).

**Verification strategy**: executed in two steps — first modify only `mac_unit` without touching `addr_pipe` and confirm the testbench **fails broadly** (proving the tests are latency-sensitive); then add `addr_pipe3` and confirm all tests pass. Both checkpoints have a definite expected outcome.

### 4.5 V6: `c_mem` Write-Enable Gating

`c_mem` was originally written every cycle (`we = mac_valid_out`), so each output element was written nine times, the first eight being intermediate partial sums that get overwritten. Changing this to `we = mac_valid_out && final_pipe3` reduced write activity by 89%.

**Result**: switching power −4.6%, total power −1.9%. Cost: +46 cells and 18 slew violations against the self-imposed 1.2 ns target (still 0 against the library's 1.5 ns limit). At 4×4, `c_mem` holds only 80 flip-flops so the benefit is modest; the change is considerably more valuable in the 7×7 configuration (500 flip-flops).

### 4.6 Migration of the Critical Path

| Version | #1 critical path | Length |
|---|---|---|
| V2 | `k` counter → `rearranged` 36:1 read mux → `mac_a` | 9.56 ns |
| V3 | Same (mux reduced to 16:1) | 9.51 ns |
| V4 | `MAC.b` → 8×8 multiplier → `product_reg` | 9.80 ns |
| V5 / V6 | `MAC.a` → **`pp_hi` partial product** → register | 7.83 ns |
| Final | Same | **6.81 ns** |

**Each optimization pushed the bottleneck onto the next structure** — which is why the STA report must be re-read after every change rather than reusing the previous conclusion.

---

## 5. Physical Design (PD) Optimization

RTL frozen at V6, clock period fixed at 8.25 ns. **No functional re-verification required.**

| Step | Change | Cells | Die µm² | Util | Setup slack | Diodes |
|---|---|---|---|---|---|---|
| Baseline | `th=86, util=30, AREA 0` | 4,549 | 77,470 | 42.3% | +0.4004 | 1,524 |
| PD-a | `th=120` | 4,318 | 77,470 | 41.4% | +0.4015 | 1,293 |
| PD-b | `th=150, util=35/40` | 3,732 | 67,026 | 46.2% | +0.5181 | 863 |
| PD-c | `th=180, util=40/45` | **3,251** | **59,159** | 50.4% | +0.3817 | **495** |
| PD-d | `SYNTH_STRATEGY = DELAY 3` | 3,547 | 63,095 | 50.8% | **+1.6018** | 579 |

### 5.1 Antenna Threshold Retuning (`HEURISTIC_ANTENNA_THRESHOLD` 86 → 180)

**Evidence**: the value `86` had been bracketed on the **previous architecture** (die 144,584 µm², wirelength 176,739 µm). The V3 implicit-im2col change reduced wirelength to 58,982 µm — **one third of the original** — making that absolute threshold far too aggressive: 1,524 diodes accounted for 33.5% of all cells.

**Result**: raising the threshold to 180 reduced diodes by **67.5%** (1,524 → 495) and cell count by 28.5%, while `antenna__violating__nets` remained **0** throughout.

> **Methodological note**: architectural changes invalidate previously tuned PD parameters. Any parameter bracketed against "design size" (antenna threshold, fanout constraint, utilization) must be re-evaluated after the architecture changes.

### 5.2 Utilization (`FP_CORE_UTIL` 30 → 40)

At step PD-a, cell count fell 5.1% but **die area remained fixed at 77,470 µm²** — the saving appeared only as reduced utilization. Raising `FP_CORE_UTIL` converted the area saving into an actually smaller die: **77,470 → 59,159 µm² (−23.6%)**.

Cost: setup slack fell from +0.5181 to +0.3817 (−0.14 ns).

### 5.3 Synthesis Strategy (`SYNTH_STRATEGY` AREA 0 → DELAY 3)

OpenLane's built-in `SynthesisExploration` flow (minutes, synthesis only) was used to sweep all nine strategies:

| Strategy | Gates | Cell area | Synthesis-stage ss slack | vs baseline |
|---|---|---|---|---|
| **`AREA 0`** (default) | 1,569 | 20,591 | +0.0922 | — |
| `AREA 1` | 1,573 | 20,611 | +0.6539 | +0.56 |
| `AREA 2` | 1,560 | **20,357** | +0.1954 | +0.10 |
| `AREA 3` | 2,456 | 24,290 | **+1.5395** | +1.45 |
| `DELAY 0` | 1,720 | 22,459 | +0.8716 | +0.78 |
| `DELAY 1` | 1,747 | 22,552 | **−0.9997** | −1.09 |
| `DELAY 2` | 1,729 | 22,359 | −0.2164 | −0.31 |
| **`DELAY 3`** | 1,681 | 22,061 | **+1.2412** | **+1.15** |
| `DELAY 4` | 2,162 | 24,139 | +0.9374 | +0.85 |

`DELAY 3` was selected: second-best slack improvement, but at a far lower area cost (+7.1%) than `AREA 3` (+18%).

**Full-flow verification**: setup slack **+0.3817 → +1.6018 (+1.22 ns)**, at a cost of +6.7% die area and +1.2% power, with all hard signoff checks remaining at zero.

> **Caveat**: synthesis-stage slack is estimated using wire-load models and diverges significantly from post-PnR results (the baseline estimated +0.0922 but measured +0.3817). `DELAY 1` and `DELAY 2` even report negative values. This table is usable **only for relative ranking**; the final answer must come from a complete flow run. In this case the exploration predicted +1.15 ns and the measurement gave +1.22 ns — the relative ranking was accurate.

### 5.4 Standard Cell Library Swap (not executable)

Section 3.2 of the course optimization guide recommends trying `sky130_fd_sc_hs`. OpenLane failed during PDK resolution:

```
_tkinter.TclError: no files matched glob pattern
"/opt/basics/pdks/sky130A/libs.ref/sky130_fd_sc_hs/lef/*.lef"
```

The course sky130A installation contains only `sky130_fd_sc_hd`, `sky130_fd_sc_hdll` and `sky130_fd_sc_hvl` under `libs.ref/` (plus io / pr / sram_macros). **There is no hs / ms / ls library.**

The only available alternative core-logic library is `sky130_fd_sc_hdll` (high density, low leakage — high-threshold-voltage devices). Analysis shows it is inapplicable to this design: **leakage accounts for only 0.0005% of total power (34 nW of 6.74 mW)**, so the leakage improvement it offers is meaningless while the delay penalty is real. This optimization therefore cannot yield a useful result in the current environment.

---

## 6. Clock Period Sweep (DELAY 3 / th=180 / util=40)

| Period | Fmax | Setup slack | Hold slack | Power | Cells | Die | Result |
|---|---|---|---|---|---|---|---|
| 8.25 ns | 121.2 MHz | +1.6018 | +0.1105 | 5.613 mW | 3,547 | 63,095 | ✓ comfortable |
| 7.25 ns | 137.9 MHz | +0.7668 | +0.1082 | 6.300 mW | 3,553 | 63,095 | ✓ |
| 7.00 ns | 142.9 MHz | +0.2640 | +0.1110 | 6.505 mW | 3,535 | 63,095 | ✓ practical margin |
| **6.85 ns** | **146.0 MHz** | **+0.0418** | +0.1089 | 6.736 mW | 3,540 | 63,095 | ✓ **final** |
| ~6.81 ns | 146.8 MHz | 0 | — | — | — | — | theoretical floor |

**Two observations**:

1. **Area is unchanged throughout** (die 63,095 µm², cell count varying by ±18). This 1.4 ns was purchased purely from timing margin at no area cost, indicating the design remained far from the congestion limit across this range.

2. **Power scales precisely linearly with frequency**: frequency ×1.205, power ×1.200.

**On the 6.85 ns margin**: +0.0418 ns (42 ps) falls within the run-to-run variation of place-and-route. Industrial practice typically requires 5–10% of the period as margin (340–680 ps here). Accordingly:
- **7.00 ns / 142.9 MHz** is the recommended configuration with practical engineering margin
- **6.85 ns / 146.0 MHz** is the measured limit

### 6.1 An Important Observation About Energy

Computing energy per 4×4 convolution (42 cycles) at the two endpoints of the sweep:

```
8.25 ns : 346.5 ns × 5.613 mW = 1.945 nJ
6.85 ns : 287.7 ns × 6.736 mW = 1.938 nJ      difference: 0.4%
```

**Raising the clock frequency alone barely changes the energy per operation** — the work finishes sooner but power rises proportionally. Energy per operation can only be reduced by **architectural changes** (fewer cycles, less area).

This explains the project's energy accounting:

```
Start  (13 ns, 48 cycles, 6.56 mW) : 624 ns × 6.56 mW = 4.09 nJ
Final  (6.85 ns, 42 cycles, 6.74 mW): 288 ns × 6.74 mW = 1.94 nJ    −52.6%
```

Essentially all of the saving came from the 2.17× reduction in time, with power almost unchanged (+2.7%). The time reduction in turn came from a 12.5% reduction in cycle count (implicit im2col eliminating the IM2COL phase) and a critical path reduction from 11.75 ns to 6.81 ns (the combined effect of four RTL changes and the PD tuning).

---

## 7. Documented Failures

| Parameter | Value | Result | Mechanism |
|---|---|---|---|
| `CLOCK_PERIOD` | 8.0 ns (default hold margin) | hold **−0.050**, 6 violations | Setup left only +0.205, giving the resizer no room to insert hold-fixing delay cells |
| `CLOCK_PERIOD` | 8.0 ns (hold margin 0.20/0.15) | setup **−0.160**, 3 violations | Fixing hold required 191 additional buffers → cell area +6.7% → setup lost 0.365 ns |
| `MAX_FANOUT_CONSTRAINT` | 6 (7×7 variant) | setup **−2.995**, 284 violations | 1,436 additional buffers → wirelength +11% → **over-buffering becomes self-defeating** (worst fanout actually degraded from 22 to 27) |
| `FP_CORE_UTIL` | 40 (7×7 variant) | **GRT-0118 congestion, flow aborted** | Die shrank while cell count grew simultaneously. Guide §3.5 states that congestion calls for *lowering* utilization |
| `DESIGN_REPAIR_MAX_WIRE_LENGTH` | 100 µm | `RSZ-0065` warning | The critical wire length for this technology is **6335 µm**; below that, inserting buffers *increases* delay. The design's longest nets are only a few hundred µm, so this repair does not apply |
| `STD_CELL_LIBRARY` | `sky130_fd_sc_hs` | PDK resolution failure | Library not installed (see §5.4) |

**The two 8.0 ns runs together form a complete demonstration**: the same clock period, changing only the resizer's hold margin, flips the design between two distinct failure modes. This quantifies the phenomenon that setup and hold repair compete for the same placement resources near the timing limit.

---

## 8. Parallel MAC Variant (7×7 / 5 MACs)

Separate branch; source file `src/cnn_top.v` (parallel version).

### 8.1 Architecture

Unrolls the **window loop** (rather than the k loop of the dot product), with parallelism `P = OW = W − 2`:

```
                  ┌─ a_mem[fmap_addr_reg + 0] ─→ MAC0 ─→ acc0 ─→ c_mem[orow·P+0]
B_MEM[k] ──┬─────→├─ a_mem[fmap_addr_reg + 1] ─→ MAC1 ─→ acc1 ─→ c_mem[orow·P+1]
(broadcast)├─────→├─ ...
           └─────→└─ a_mem[fmap_addr_reg+P-1] ─→ MAC(P-1) ────→ c_mem[orow·P+P-1]
```

**Three reasons for choosing `P = OW`**:

1. In the address decomposition `addr = row_base + tap_off + ocol`, the term `ocol` **is exactly the MAC index**, so the P addresses within any cycle are naturally consecutive
2. Each batch processes one complete row of output windows, so `row_base` advances by `+W` per batch (eliminating the in-row `+1` / row-boundary `+3` decision logic)
3. `M = OH × OW` always holds, so **there is no tail batch**

**Weight-stationary dataflow**: all MACs consume the same kernel tap in any given cycle, so `B_MEM` needs only one read port with a broadcast; `mac_valid` and `mac_clear` are shared. Because the P MACs are perfectly synchronized, pipeline tracking still requires only **one** `addr_pipe` chain.

Unrolling the k loop instead would require a different kernel value per MAC (P read ports on `B_MEM`), and `K = 9` does not divide evenly by typical values of P.

### 8.2 Results

| Configuration | ns | MHz | FF | Cells | Die µm² | mW | slew (lib) | ant | setup |
|---|---|---|---|---|---|---|---|---|---|
| th=86, fo=20 | 12.0 | 83.3 | 1,382 | 29,398 | 355,509 | 14.45 | 904 | 2 | +1.240 |
| th=200, fo=20 | 12.0 | 83.3 | 1,382 | 24,850 | 355,509 | 14.24 | 807 | 6 | +1.285 |
| **th=200, fo=10** | **13.0** | **76.9** | 1,382 | **23,171** | 355,509 | **12.87** | **56** | 9 | +1.212 |

**Best configuration**: 13 ns / 76.9 MHz, die **590.9 × 601.6 µm = 355,509 µm²**, one 7×7 convolution in **51 cycles = 663 ns**.

### 8.3 Fair Comparison Across Input Sizes

Absolute times for different input sizes are not directly comparable; normalizing per output element gives a fair basis (every version performs nine multiply-accumulates per output element):

| Version | Output elements | ns / output | nJ / output |
|---|---|---|---|
| V2 starting point (4×4 @ 13 ns) | 4 | 156.0 | 1.023 |
| **Final single-MAC (4×4 @ 6.85 ns)** | 4 | **72.0** | **0.485** |
| **Parallel 5-MAC (7×7 @ 13 ns)** | 25 | **26.5** | **0.341** |

**The parallel variant is 2.7× faster and 30% more energy-efficient per output element** than the final single-MAC version, despite a lower Fmax (76.9 vs 146.0 MHz) and 5.6× the die area. This is the most direct quantification of parallelism improving throughput — and it also demonstrates that **Fmax is a misleading metric when evaluating parallel architectures; time and energy per unit of work are the correct measures.**

Cycle count comparison (7×7):

| | COMPUTE | Total |
|---|---|---|
| Single MAC (projected) | M×K = 225 | ~231 |
| **5 MACs (measured)** | OH×K = **45** | **~51** |

### 8.4 Outstanding Issues (recorded honestly)

The 7×7 variant retains **56 slew violations against the library's 1.5 ns limit** and **9 antenna violations** that could not be eliminated at the configuration level.

**Root cause analysis**: the violating pins cluster on `a221o_1` / `mux2_1` / `a22o_1` cells; tracing the netlist confirmed these belong to the internal structure of two very large read multiplexers — the 5 × 49:1 on `a_mem` and the 25:1 × 20-bit on `c_mem`. Spanning a 591 µm die, these nets are long and moderately fanned out, and minimum-drive cells cannot meet the transition-time requirement.

The `RSZ-0065` warning simultaneously confirms this is **not a wire-delay problem** (the technology's critical wire length is 6335 µm, far longer than any net in this design) but a **drive-strength versus load** problem.

**Tried and effective**: `MAX_FANOUT_CONSTRAINT` 20 → 10, reducing library-limit slew violations from **807 to 56 (−93%)**.
**Tried and counterproductive**: tightening further to 6, which collapsed setup timing through over-buffering (see §7).

**Root-cause fixes (not implemented)**:
- Bank `a_mem` into 5 banks by `addr mod P` (10 entries each), reducing read mux depth from 49:1 to 10:1. Costs a P-way barrel rotator, and since `P = 5` is not a power of two, `addr mod 5` / `addr div 5` must be maintained as counters
- Add an output register to the `c_mem` read path, splitting the 25:1 × 20-bit combinational path that currently runs straight to the output pins. Costs a change to the interface contract (combinational read becomes registered read, requiring a corresponding testbench change)

---

## 9. Methodological Notes

Several observations from this project that generalize:

**9.1 `Flow complete` ≠ clean signoff**
In OpenLane, setup, hold, max slew, max cap and antenna checks are **all warning-level** and do not halt the flow. The `ERROR_ON_*` family of switches covers only DRC / LVS / XOR / illegal overlap / disconnected pins / PDN / unmapped cells. Determining whether a design closed requires reading `metrics.csv`; the flow completing is not evidence. This project went several rounds without noticing slew violations for exactly this reason.

**9.2 Present in `config.yaml` ≠ in effect**
Three distinct classes of silently-ineffective configuration were found:

| Symptom | Cause |
|---|---|
| `MAX_FANOUT_CONSTRAINT` and two other electrical constraints had no effect | These variables are delivered through the SDC template; a custom `PNR_SDC_FILE` overrode the default `base.sdc`, discarding its `set_max_*` statements |
| `PL_RESIZER_DESIGN_OPTIMIZATIONS` and three others had no effect | OpenLane **1** variable names, unrecognized by OpenLane 2 |
| `RUN_ANTENNA_REPAIR: true` was redundant | Already true by default |

The first was the most damaging: `repair_design` is a constraint-driven step, and with no constraints in the SDC it had nothing to fix — across six consecutive runs it inserted only 68 buffers, leaving a net with fanout 130 unsplit. After the fix (reverting to the default `base.sdc`), buffer insertion rose to 618–1,130.

**The only reliable verification** is to inspect `runs/<run>/resolved.json` and `runs/<run>/final/sdc/*.sdc` after the run.

**9.3 The two slew criteria must be reported separately**
`design__max_slew_violation__count` counts violations against the **SDC self-imposed constraint** (1.2 ns in this project); the actual signoff criterion is the **standard cell library's characterization limit** (1.5 ns for `sky130_fd_sc_hd`). The two can differ by an order of magnitude — one 7×7 run reported 1,558 versus 807. Setting a self-constraint tighter than the library limit is good practice (it drives optimization), but reports must state which criterion is being used.

**9.4 The gap to the next path determines whether an optimization is worthwhile**
When several critical paths are clustered within a narrow range, the maximum benefit from improving any single path is the gap to the next-worst path. In this project, the expected payoff from pipelining the multiplier was +7%, +2.4% and +35% at three different points in time, depending entirely on where the second-worst path sat. **The STA report must be re-read after every change; the previous conclusion cannot be reused.**

**9.5 Over-optimization is self-defeating**
The same pattern was observed repeatedly: past a certain buffer density, the buffers' own delay and load exceed what they save. See §7 for representative cases. The diagnostic is abnormal growth in `timing_repair_buffer` and `wirelength`.

**9.6 Place-and-route is deterministic**
Identical RTL plus identical configuration reproduces bit-identical results. After the source file for the 8.25 ns version was lost, it was reconstructed from a backup and verified by confirming that **all 22 metrics matched exactly** (down to `cell area 28882.7` and `power 5.7157 mW`). This provides an objective criterion for whether a reconstruction is correct.

---

## 10. Data File Index

Mapping of metrics files in `results/` to experiments:

### Single-MAC main line

| File | Version | Period | Notes |
|---|---|---|---|
| `clk13_after_aidx.csv` | V2 | 13.0 ns | `a_idx` counter |
| `clk11_implicit_im2col.csv` | V3 | 11.0 ns | Implicit im2col |
| `clk10.5_repipelining.csv` | V4 | 10.5 ns | Address precomputation |
| `clk10_mac3stage.csv` | V5 | 10.0 ns | Three-stage MAC |
| `clk8.5_mac3stage.csv` | V5 | 8.5 ns | Clock sweep |
| `clk8.25_FINAL.csv` | V5 | 8.25 ns | Clock sweep |
| `clk8.25_FINAL_wegated.csv` | **V6** | 8.25 ns | **PD tuning baseline** |

### PD tuning and final sweep

| File | Configuration | Notes |
|---|---|---|
| `8.25_th120_util30.csv` | th=120 / util=30 | PD-a |
| `8.25_th150_util35.csv` | th=150 / util=35 | PD-b |
| `8.25_th180_util40.csv` | th=180 / util=40 | PD-c |
| `8.25_th180_util40_DELAY3.csv` | + DELAY 3 | PD-d |
| `7.25_DELAY3.csv` | 7.25 ns | Clock sweep |
| `7.0_DELAY3.csv` | 7.0 ns | Recommended (practical margin) |
| **`6.85_DELAY3.csv`** | **6.85 ns** | **★ Final** |

### Failures and parallel variant

| File | Contents |
|---|---|
| `clk8.0_HOLDFAIL.csv` | 8.0 ns, hold failure |
| `clk8.0_SETUPFAIL_holdmargin.csv` | 8.0 ns with raised hold margin, setup failure |
| `clk12_parallel_mac.csv` | 7×7 / 5 MACs @ 12 ns |

---

## Appendix A: Reproduction

### A.1 Functional Verification

```bash
# Single size
iverilog -g2012 -o sim src/tb_cnn_top.v src/cnn_top_0825.v src/mem.v src/mac_unit.v
vvp sim

# Full size sweep
for n in 3 4 5 6 7; do
  iverilog -g2012 -Ptb_cnn_top.NDIM=$n -o sim_$n \
           src/tb_cnn_top.v src/cnn_top_0825.v src/mem.v src/mac_unit.v
  vvp sim_$n > log_$n.txt
  echo -n "NDIM=$n: "; grep RESULT log_$n.txt
done
```

Expected: `RESULT: PASS` for all sizes. At NDIM=4 this is 43 tests and 172 value checks.

### A.2 Physical Implementation

Final configuration file: `config_0825.yaml` (with `VERILOG_FILES` pointing at `src/cnn_top_0825.v`)

```yaml
CLOCK_PERIOD: 6.85
FP_CORE_UTIL: 40
PL_TARGET_DENSITY_PCT: 45
MAX_FANOUT_CONSTRAINT: 20
MAX_TRANSITION_CONSTRAINT: 1.2
MAX_CAPACITANCE_CONSTRAINT: 0.25
SYNTH_STRATEGY: "DELAY 3"
RUN_HEURISTIC_DIODE_INSERTION: true
HEURISTIC_ANTENNA_THRESHOLD: 180
GRT_ANTENNA_MARGIN: 20
GRT_ANTENNA_ITERS: 20
DIODE_PADDING: 2
RUN_POST_GPL_DESIGN_REPAIR: true
RUN_POST_GRT_DESIGN_REPAIR: true      # defaults to false — must be enabled explicitly
RUN_POST_CTS_RESIZER_TIMING: true
RUN_POST_GRT_RESIZER_TIMING: true
STD_CELL_LIBRARY: "sky130_fd_sc_hd"
```

**Important**: do not set `PNR_SDC_FILE` or `SIGNOFF_SDC_FILE`. Use OpenLane's default `base.sdc`, otherwise the three `MAX_*_CONSTRAINT` values never reach the tools (see §9.2).

```bash
openlane --manual-pdk --pdk sky130A --pdk-root "$PDK_ROOT" config_0825.yaml
```

### A.3 Result-Checking Script

```bash
#!/bin/bash
# check.sh — verify all signoff items for the most recent run
N=$(ls -dt runs/RUN_* | head -1)
echo "=== $N ==="
python3 - "$N" <<'EOF'
import csv, sys, glob
d = {r[0]: r[1] for r in csv.reader(open(sys.argv[1] + '/final/metrics.csv')) if len(r) == 2}
must_zero = ['timing__setup_vio__count', 'timing__hold_vio__count',
             'design__max_cap_violation__count', 'antenna__violating__nets',
             'magic__drc_error__count', 'klayout__drc_error__count',
             'route__drc_errors', 'design__lvs_error__count',
             'design__xor_difference__count', 'flow__errors__count']
bad = 0
for k in must_zero:
    v = d.get(k, '?')
    ok = v != '?' and float(v) == 0
    bad += not ok
    print(f"{'OK ' if ok else 'FAIL'}  {k:44s} {v}")

# max slew must be evaluated against both criteria
f = glob.glob(sys.argv[1] + '/*-openroad-stapostpnr/max_ss_100C_1v60/checks.rpt')
if f:
    t = open(f[0]).read(); o = []
    for L in t.split('max slew')[1].split('max fanout')[0].splitlines():
        q = L.split()
        if len(q) >= 4:
            try: o.append(float(q[2]))
            except: pass
    lib = sum(1 for v in o if v > 1.5)
    bad += lib > 0
    print(f"{'OK ' if lib == 0 else 'FAIL'}  max slew vs library limit 1.5 ns{'':12s} {lib}")
    print(f"      max slew vs SDC constraint 1.2 ns{'':11s} {d.get('design__max_slew_violation__count')}")

for k in ['timing__setup__ws', 'timing__hold__ws', 'design__die__area',
          'design__instance__count__stdcell', 'design__instance__utilization',
          'power__total']:
    print(f"      {k:44s} {d.get(k)}")
print('\n=== VERDICT:', 'CLEAN' if bad == 0 else f'{bad} ITEM(S) NOT CLEAN', '===')
EOF
```

### A.4 Source File Versions

| File | Contents |
|---|---|
| **`src/cnn_top_0825.v`** | **V6, single MAC, final version** |
| `src/cnn_top.v` | Parallel variant (7×7 / 5 MACs) |
| `src/cnn_top_v2_implicit_im2col.v.bak` | V4 (address precomputation, two-stage MAC) |
| `src/mac_unit.v` | Three-stage pipelined MAC, shared by all versions |
| `src/mem.v` | Generic register array (combinational read) |
| `src/tb_cnn_top.v` | Self-checking testbench |
| `config_0825.yaml` | OpenLane configuration for the single-MAC version |
| `config.yaml` | OpenLane configuration for the parallel variant |
