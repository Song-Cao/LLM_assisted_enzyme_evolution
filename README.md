# CT-Contrastive: a contrastive LLM approach to guide the directed evolution of enzymes
--------

Enzyme evolution is far from a solved problem --- 3D diffusion/MF-based method can only hallucinate based on known motif scaffolds & residues, but fail to optimize/diversify enzyme-substrate contacting pose & reactions; sequence-based approaches can optimize enzyme residues based on fitness data, but fail to utilize/correct data from different evolutionary batch/round. Here we design a contrastive protein LLM approach to tackle these challenges by learning the granularity between evolutionary events/mutations along the directed evolution trajectory, and utilized the pipeline to suggest mutations for a recently discovered metallo-esterase [Studer et al. Science](https://www.science.org/doi/10.1126/science.aau3744)

![CT-Contrastive](dual-mode-fitness-prediction.png)

## Environment setup
## Running scripts
### Data preprocessing:
  - Utilize `Data_preprocessing/fitness_cal.py` for read merging and fitness calculation
  - Utilize `Data_preprocessing/combinability.py` for mutational combinability and mutability analysis
### Dataset construction:
  Refer to `scripts/data_construct.ipynb` for 
  - noise filtering
  - AAIndex/ESM2/PLL embeddings
  - train/test/inference dataset construction
### Model training
  - For training the single prediction model and conduct some preliminary evaluations, run `train_SingleRR.ipynb`
  - For training our CT-Contrastive and other contrastive models, configure your model and run `train_Contrastive.ipynb`
### Model evaluations
  For model evaluations including suggestion rank based correaltion and top N suggestion overlap, please refer to `Compare_models.ipynb`
### Inference/ Mutation Suggestion
  Refer to `scrips/Suggest_mutations...ipynb`
