# CASSI-VegIndex

Direct multi-index vegetation estimation from CASSI compressed measurements.

The project maps a two-dimensional CASSI measurement to 32 dense vegetation-index
maps without requiring full hyperspectral reconstruction as an intermediate task.
It currently includes a Restormer placeholder baseline and four candidate models:
MST-Mamba, WPO3D, DHM, and IFGNet.

## Repository layout

- `code/` -- active training, testing, visualization, models, and technical reports.
- `latex/zh.tex` -- Chinese teacher-review manuscript.
- `latex/eng.tex` -- preserved English draft snapshot.
- `latex/logic.md`, `algorithm.md`, `citation.md` -- manuscript support notes.

Large datasets, checkpoints, experiment outputs, source backups, historical code,
and third-party reference implementations are intentionally excluded from Git.
They remain on the research server under `/data5/SCI/vegindex`.

## Current status

- 252 paired HSI/index scenes and one physical mask are available server-side.
- Restormer completed 300 training epochs; its epoch-300 file is a
  training-loss-selected placeholder, not a validation-selected best model.
- All five model interfaces pass a CPU-only synthetic forward smoke test.
- Legacy quantitative test results are invalid because the former test script
  normalized model outputs twice; corrected GPU evaluation is still pending.

See [the code README](code/README.md),
[the evaluation report](code/docs/evaluation_report.md), and
[the directory map](code/docs/directory_map.md) for details.
