# Latent Space Model Analysis Report

**Generated**: 2026-02-19 08:23
**Model**: latent_model_20260219_082341

## Training Results

- **Best Epoch**: 4
- **Best Val Loss**: 7.9117
- **Test RMSE**: 1.734259843826294
- **Test Correlation**: -0.004761555491128021

## Discovered Archetypes

### Low Games Played Type (Archetype 0)

- **Players**: 68
- **Avg RAPM**: 0.16
- **Survival Rate**: 75.0%
- **Description**: Characterized by low Games Played
- **Examples**: Jalen Johnson, Josh Hall, Justin Robinson

### Low Shot Dist Var Type (Archetype 1)

- **Players**: 115
- **Avg RAPM**: 0.37
- **Survival Rate**: 62.6%
- **Description**: Characterized by low Shot Dist Var
- **Examples**: Julian Champagnie, Immanuel Quickley, Rui Hachimura

### High Shot Dist Var Type (Archetype 2)

- **Players**: 78
- **Avg RAPM**: 0.00
- **Survival Rate**: 62.8%
- **Description**: Characterized by high Shot Dist Var
- **Examples**: Chet Holmgren, Jaden Ivey, Keon Ellis

### Low Shot Dist Var Type (Archetype 3)

- **Players**: 60
- **Avg RAPM**: -0.13
- **Survival Rate**: 73.3%
- **Description**: Characterized by low Shot Dist Var
- **Examples**: Jericho Sims, Marcus Garrett, Cole Anthony

### High Minutes Total Type (Archetype 4)

- **Players**: 75
- **Avg RAPM**: 0.29
- **Survival Rate**: 66.7%
- **Description**: Characterized by high Minutes Total
- **Examples**: Marvin Bagley III, Wendell Carter Jr., Gary Trent Jr.

### High Avg Shot Dist Type (Archetype 5)

- **Players**: 91
- **Avg RAPM**: -0.12
- **Survival Rate**: 64.8%
- **Description**: Characterized by high Avg Shot Dist
- **Examples**: Andrew Nembhard, Johnny Juzang, Jalen Duren

### Low Season Type (Archetype 6)

- **Players**: 86
- **Avg RAPM**: 0.05
- **Survival Rate**: 70.9%
- **Description**: Characterized by low Season
- **Examples**: Mark Williams, James Wiseman, Udoka Azubuike

### Low Shot Dist Var Type (Archetype 7)

- **Players**: 52
- **Avg RAPM**: 0.24
- **Survival Rate**: 67.3%
- **Description**: Characterized by low Shot Dist Var
- **Examples**: Peyton Watson, Admiral Schofield, Cameron Johnson

### High Ft Att Type (Archetype 8)

- **Players**: 143
- **Avg RAPM**: 0.51
- **Survival Rate**: 74.8%
- **Description**: Characterized by high Ft Att
- **Examples**: AJ Griffin, Brandon Clarke, Coby White

### Low Games Played Type (Archetype 9)

- **Players**: 42
- **Avg RAPM**: -0.03
- **Survival Rate**: 61.9%
- **Description**: Characterized by low Games Played
- **Examples**: Christian Koloko, Malik Fitts, Mfiondu Kabengele

### High Avg Shot Dist Type (Archetype 10)

- **Players**: 79
- **Avg RAPM**: -0.41
- **Survival Rate**: 55.7%
- **Description**: Characterized by high Avg Shot Dist
- **Examples**: Jamorko Pickett, Collin Gillespie, Lester Quinones

### High Xy Rim Shots Type (Archetype 11)

- **Players**: 27
- **Avg RAPM**: -0.08
- **Survival Rate**: 74.1%
- **Description**: Characterized by high Xy Rim Shots
- **Examples**: Omer Yurtseven, Kenneth Lofton Jr., Jalen Williams

## Usage

```python
from models import ProspectModel, ArchetypeAnalyzer

# Load model
model = ProspectModel(latent_dim=32, n_archetypes=8)
model.load_state_dict(torch.load('model.pt'))

# Analyze a prospect
analysis = analyzer.analyze_player(tier1, tier2, career, tier2_mask)
print(analysis.narrative)
```
