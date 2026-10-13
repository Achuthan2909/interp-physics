# Literature Matrix

| Question | Paper | Takeaway |
|----------|-------|----------|
| Is the probe just memorising? | [Hewitt & Liang 2019, control tasks](notes/hewitt-2019-control-tasks.md) · [PDF](pdf/probing/1909.03368v1.pdf) | Compare against a fake task; prefer linear probes |
| Is the information just the inputs passed through? | [Hewitt et al. 2021, conditional probing](notes/hewitt-2021-conditional-probing.md) · [PDF](pdf/probing/2109.09234v1.pdf) | Report what a layer adds beyond a baseline representation |
| Would any network of this shape score the same? | [Pimentel et al. 2020, control functions](notes/pimentel-2020-info-theoretic-probing.md) · [PDF](pdf/probing/2004.03061v2.pdf) | Compare against a randomised or untrained representation |
| How much effort does the probe need? | [Voita & Titov 2020, MDL probing](notes/voita-2020-mdl-probing.md) · [PDF](pdf/probing/2003.12298v1.pdf) | Easy-to-extract information is stronger evidence than hard-to-extract |
| Which probe is the right one? | [Pimentel et al. 2022, architectural bottleneck](notes/pimentel-2022-architectural-bottleneck.md) · [PDF](pdf/probing/2211.06420v1.pdf) | Match the probe's form to how the next layer reads the representation |
| Does the score survive new data? | [Levinstein & Herrmann 2023–24](notes/levinstein-2023-no-lie-detector.md) · [PDF](pdf/probing/2307.00175v1.pdf) | A probe can score well and still fail on simple variants |
| Is it detecting a correlated feature instead? | ["Linear probes rely on textual evidence" 2025](notes/boxo-2025-linear-probes-textual-evidence.md) · [PDF](pdf/probing/2509.21344v2.pdf); ["Linear Probes Detect Task Format, Not Reasoning Mode" 2026](notes/sahoo-2026-probes-task-format.md) · [PDF](pdf/probing/2606.02907v2.pdf) | Control for confounds before claiming the concept is encoded |
| Does the network use it? | [Elazar et al. 2021, amnesic probing](notes/elazar-2020-amnesic-probing.md) · [PDF](pdf/probing/2006.00995v3.pdf) | Probe score does not predict importance to behaviour |
| Was the removal clean? | ["How Reliable are Causal Probing Interventions?" 2025](notes/canby-2024-causal-probing-reliability.md) · [PDF](pdf/probing/2408.15510v5.pdf) | Check the target is gone and other properties are intact |
| Overview |[Belinkov 2022 survey](notes/belinkov-2021-probing-survey.md) · [PDF](pdf/probing/2102.12452v4.pdf) | The framing reference for a probing chapter |
