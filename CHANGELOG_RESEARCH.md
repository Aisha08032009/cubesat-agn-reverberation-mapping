# Computational audit and model revisions

The original Python source is preserved without edits in `original_source/original_python_scripts.zip`.

- **OU/DRW amplitude corrected:** Original initialized SD ~0.20 but used a stationary innovation process with SD ~0.04 for tau=50 and sf_inf=0.20. Corrected definition uses `SF_inf = sqrt(2) * SD_stationary`. This is not a numerical reproduction of the manuscript, because the physical input model changed.
- **Echo convolution edge handling corrected:** Original `np.convolve(..., mode="same")` implicitly zero-padded and created artificial endpoint dimming. Corrected code edge-extends before filtering. The first ~12 days still lack pre-campaign source history; future work should simulate burn-in.
- **Full-constraint FR/RSS setting:** The original script used 300 FR/RSS trials even though the manuscript discussed 1,200; the research script is now set to 1,200. Outputs must be regenerated. The original is retained in the archive.
- **Target coordinate consistency:** The original full-constraint model uses `beta=55°` for Earth occultation but an independently chosen ecliptic (lon=340°, lat=42°) direction for Sun avoidance, so these are not one physical sky target. By default, the updated full-constraint model uses the ecliptic position to determine both; `consistent_target_geometry=False` reproduces the legacy geometry. This changes absolute visibility and uncertainty and is NOT a manuscript reproduction.
- **Comparison caveat:** Legacy comparison scripts no longer hardcode manuscript baseline percentages; they load newly computed `results.json` if available, otherwise omit the comparison.
- **Validation status:** A fast diagnostic run is supplied separately in `results/`; full manuscript-scale Monte Carlo runs are expensive and must be identified by their actual seeds and trial counts. Paper table values belong to the archived historical manuscript, not to corrected-run outputs.

## Additional scientific concerns

- The original FR/RSS uses unique resampled indices, so duplicate draws are dropped; this follows one simplified convention, but alternative methods weight retained epochs differently.
- ICCF centroids are calculated across **all** grid points exceeding 80% of the peak, not just the connected region around the primary peak. Multimodal CCFs can yield misleading centroids.
- The 120-day simulation starts without any earlier continuum history; even with corrected boundary padding, a physical reverberation signal needs a pre-window burn-in to eliminate transient assumptions.
- The discrete observation cadence (`2 days / N`) is a design assumption, not automatically a schedule achievable by every orbital geometry or instrument.
- Welch t-tests on per-realization uncertainty may not be independent when the different configurations use distinct random realizations; a paired-seed design can improve control of nuisance variability.
- Results from new corrected simulations should not replace values in the original paper without updating that paper and a method-by-method comparison.
