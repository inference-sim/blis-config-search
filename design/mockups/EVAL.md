# Gallery mockup bake-off: evaluation

Judge method: read all five files; ran each file's filter, sort, exec-pick and scale logic under node against `data.js`; rendered all four in headless Chrome (1300px desktop, dark scheme) and probed a 390px iframe for horizontal overflow, `@import` validity and a live click. I did not exercise light mode visually. I checked it statically from the `:root` tokens.

## Scores (0-5)

| | R1 filter+counts | R2 exec pick | R3 report links | R4 how/PR/trust | R5 mock marking | R6 self-contained, light+dark, responsive | R7 design fidelity | Total /35 |
|---|---|---|---|---|---|---|---|---|
| **A** specsheet | 3 | 3 | 5 | 3 | 3 | 3 | 3 | **23** |
| **B** table | 5 | 5 | 5 | 5 | 5 | 4 | 5 | **34** |
| **C** explorer | 5 | 4 | 5 | 5 | 4 | 4 | 3 | **30** |
| **D** brief | 5 | 4 | 5 | 4 | 4 | 5 | 4 | **31** |

## Ranking

1. **B** (34)
2. **D** (31)
3. **C** (30)
4. **A** (23)

**Ship B.** It is the only one that handles the central trust problem. With no filter the top pick is a synthetic scenario, and B says so and links the best real BLIS result right beside it. It is also the most faithful to the ../leaderboard spec-sheet language: dense, hairline-ruled, mono and tabular numbers, teal only on the best-in-column cell. Its logic had no bugs in any filter state I tried. Its weak spot is phone width, where the table scrolls sideways. Fix that and consider borrowing D's hero treatment for the exec strip.

## Verified logic results (node, against data.js)

- Unfiltered exec pick for A, B and D is `qwen3-14b-h200-summarization-0004` (265 tok/s/GPU, 1060 tok/s, 52 tok/s/user). C's default ranking agrees. This matches the expected result.
- C's alternate rankings: by tok/s the top is codegen-0006 (3120). By per-user the top is the real glm scenario (74.2).
- B: H100 plus qwen3 gives 2 rows. Facet counts are correct: H200 shows 1, glm shows 2. A zero-row filter gives `topPick(null)` and `bestIds({})` without error, and the UI handles it. A single row gives no "best" highlighting. All 10 sortable columns sort without error.
- C: `niceScale(0)` and `niceScale(NaN)` return `{max:1, step:0.2}`, so no NaN. An empty row set and a single row both give finite scales. A zero-span `linScale` returns the range midpoint.
- 390px iframe probe: no horizontal page overflow in any file. B's table scrolls inside `.tablewrap` by design. The headless screenshots at 390 look clipped only because Chrome enforces a minimum window width; this is not a real overflow.
- External URLs: only the Google Fonts `@import` in each file. All four load `data.js` as a sibling script. A real gallery build must inline it or emit it.
- **A's `@import` is invalid.** It sits at line 34, after style rules, so the CSSOM reports 0 import rules and the font request is dropped. B, C and D report 1 import rule each. The screenshot renders in Plex only because the font is installed on this machine.
- **A has a class collision.** `.opt.empty` (line 85) collides with the global `.empty{grid-column:1/-1;text-align:center;padding:60px 0}` (line 111). After clicking the H200 hardware option, every zero-count facet option measured 143px tall and centered, which breaks the rail.

## Per-file notes

### A: facet rail + card grid

- **Verdict:** a competent but generic card grid with one real layout bug and the weakest trust handling.
- **Biggest strength:** the live-count facet rail is the clearest filter UI of the four.
- **R2:** the exec shows all three numbers and recomputes under the filter. It labels a mock top pick only with plain-text "· MOCK" (line 206) and gives no pointer to a real result. The "Σ winner tok/s" cell (lines 201, 219) sums winners across different models and hardware, which is meaningless and looks authoritative.
- **R4:** one paragraph in a footer bar. It does cover the PR and CI flow and the mock note. The simulator disclaimer is in the dek.
- **R5:** cards carry a badge, but mock cards use the same teal top border as real ones (line 88). The exec marking is weak.
- **R6/R7:** dark contrast fails on `.opt.on` (white on #4bbf96, line 83). A mock-looking stat card grid with a teal top border on every card dilutes the "teal = the number to look at" rule. On phones the rail is a long vertical list that pushes the cards far down.
- **Top 3 fixes:**
  1. Move the `@import` from line 34 to the top of the `<style>` block (line 15).
  2. Rename the line 111 `.empty` rule to `.nomatch` and update line 228, so the zero-count facet options stop inheriting it.
  3. Mark the exec pick with a proper `.mock` badge at line 206 and add a "best real result" link as B does. Drop the Σ cell. Give mock cards a muted border at line 88, and use an on-accent token for `.opt.on` in dark.

### B: sortable leaderboard table

- **Verdict:** the closest to the spec-sheet language and the most honest about mock data.
- **Biggest strength:** `renderExec` lines 290-295. When the top pick is synthetic, it appends a banner with the best real BLIS result in the current view, linked.
- **R1/R2:** counts are correct and the exec recomputes. The exec adds ITL p99 and an Open report button. Column sort has aria-sort, and the expandable finalists are a good touch.
- **R5:** mock flag on every row, an amber left bar, a legend entry, a status count ("5 mock") and the exec. Best coverage of the four.
- **R6:** dark mode and breakpoints are covered, and filter chips handle dark contrast correctly. The table has `min-width:1080px` (line 111), so phones scroll sideways. There is no theme toggle, so dark follows the system setting only (the `data-theme` CSS exists but nothing sets it).
- **R7:** teal is applied to best-in-column cells and the config column, which is the intended use. It also tints the `top` row and best cells on mock rows (lines 128, 351), which gives synthetic numbers the "look here" colour.
- **Top 3 fixes:**
  1. Below about 700px, reflow rows into stacked cards instead of scrolling a 1080px table (line 111).
  2. Suppress the teal `top` and best highlighting for `synthetic` rows, or exclude mock scenarios from `topPick` (line 260) unless no real row is in view.
  3. Sub-rows put `c=${f.c_use}` under the "Servable" header (line 371). Add a "load" column or move it. Also add a theme toggle, since the data-theme CSS is already there.

### C: Pareto scatter explorer

- **Verdict:** the most ambitious and the most useful for exploration, but the plot and the exec contradict each other and one marker rule is wrong.
- **Biggest strength:** a hand-built inline-SVG scatter with real/mock encoded by marker style, plus linked hover between plot and list, and a "Data: real/mock" facet.
- **R1:** five facets with live counts, including real-vs-mock. Logic held in every state tested.
- **R2:** a rank-by toggle, with "do not deploy from it" appended for synthetic picks. But the default rank is tok/s/GPU while the plot axes are tok/s and tok/s/user, so the "TOP" ring lands on a lower-left point of a chart captioned "upper-right is better". The segmented control is clipped in the 352px panel on desktop ("tok/s/us" cut off in the screenshot).
- **R5:** marked in the plot, tooltip, list, exec and a strip. But "Show all finalists" always draws finalists as mock markers (line 460, `markerSVG(..., true)`), so the real scenario's finalists read as mock.
- **R7:** model colour uses four hues including a teal-green (`--plot-s1`, lines 21 and 33), which competes with the accent. Pressed chips use #fff on #4bbf96 in dark (lines 84-85, 154), which is low contrast.
- **Top 3 fixes:**
  1. Line 460: pass `p.s.synthetic` instead of `true`, and draw finalist polylines to match.
  2. Reconcile the plot with the ranking. Either add tok/s/GPU as marker size, or change the TOP marker and caption so a lower-left winner makes sense. Fix the clipped segmented control (lines 149-154) by letting it wrap or stack.
  3. Replace the categorical model hues with greys and shapes so teal stays reserved, and use an on-accent token for pressed chips in dark.

### D: featured deployment brief

- **Verdict:** a strong, clear answer-first layout, with some cross-scenario comparisons that mislead and the trust note placed too low.
- **Biggest strength:** the hero brief puts the config, the three metrics, the floor and the "why it won" reason in one scannable block, with the full report one click away.
- **R1/R2:** counts are correct and chips are disabled at zero. The hero recomputes under the filter. The unfiltered hero is the synthetic summarization scenario, with only an "illustrative placeholder" banner and no pointer to a real result.
- **R4:** the how-it-works and PR flow are good. The trust note is the third panel at the bottom, below the fold. The only disclaimer near the top is "Predicted at c=18 concurrent".
- **R5:** hero banner, card badges and an amber card border. No positive "real BLIS run" marker on the real scenario.
- **Trust issue:** "74% of featured tok/s/GPU" (line 286) compares across different models, hardware and workloads, and makes the one real scenario read as "30% of featured".
- **Top 3 fixes:**
  1. Feature a real scenario by default (line 203), or at least link the best real result from the mock banner at line 242.
  2. Remove or scope the "% of featured" line at 286 to same-model comparisons.
  3. Add a "Simulator predictions, not measurements" line to the masthead (line 158) so the trust note is above the fold.

## Cross-cutting findings

- **Unfiltered, the "top config to deploy" is a synthetic scenario in all four mockups.** Only B and C soften it (B links the best real result, C adds "do not deploy from it"). A and D show a bare mock. In production, mock scenarios should not appear at all.
- **"Top config to deploy" compares across different models, workloads and GPU budgets by tok/s/GPU.** That favours small models and ignores workload. It should be labelled "highest tok/s/GPU in view" unless a model and workload are selected.
- **Teal is used loosely.** Every file uses it for selected chips, CTAs, card borders and the masthead word, not only for the number to look at. B is the closest to the spec.
- **Report links follow the correct pattern in all four:** `<run_id>/index.html`, relative. They do not resolve inside `mockups/` because no run directories are there. That is expected, not a defect.
- **`run_id` is interpolated unescaped in A and B (A line 231, B line 349).** This is harmless for the mock data but should be escaped in the real build.
