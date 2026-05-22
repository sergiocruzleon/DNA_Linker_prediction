# DNA Linker Prediction

Python tools for predicting DNA linker connections between nucleosome particles from Cryo-ET tomograms.

## Install

```bash
git clone https://github.com/sergiocruzleon/DNA_Linker_prediction.git
cd DNA_Linker_prediction
pip install -e .
```

The pipeline uses CryoCAT for `.em` and `.mrc` files. Install it in the same environment before running the example if it is not already available.

To run the example notebook from a fresh environment, install the notebook tools too:

```bash
pip install -e ".[examples]"
```

## Run The Example Notebook

The repository includes a complete example under `example/`:

- `example/pipeline_config.yaml`
- `example/inputs/motl_EMD2601_dropped_01.em`
- `example/inputs/Threshold_ref_entrymask_r2_resamp_righthand.mrc`
- `example/inputs/Threshold_ref_exitmask_r2_resamp_righthand.mrc`
- `example/inputs/Threshold_ref_Origin_entrymask_r2_resamp_righthand.mrc`
- `example/inputs/Threshold_ref_Origin_exitmask_r2_resamp_righthand.mrc`

Launch Jupyter and run:

```bash
python -m notebook notebooks/run_pipeline_example.ipynb
```

Notebook outputs are written to `example/outputs/notebook/`.

## Run From The Command Line

```bash
python scripts/run_pipeline.py --config example/pipeline_config.yaml --emd 2601 --workers 1
```

To use your own input files, copy and edit `example/pipeline_config.yaml` or pass a custom config:

```bash
python scripts/run_pipeline.py --config /path/to/project_config.yaml --emd 2601 --motl-file your_particles.em
```

## More Details

Specialist configuration notes are in [docs/specialist_usage.md](docs/specialist_usage.md).

## Contact

Sergio Cruz: sn.cruz35@gmail.com
