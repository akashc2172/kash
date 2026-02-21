# Latent Space Model Analysis Report

**Generated**: 2026-02-18 20:13
**Model**: latent_model_20260218_201331

## Training Results

- **Best Epoch**: 38
- **Best Val Loss**: 2.3285
- **Test RMSE**: 1.6236587762832642
- **Test Correlation**: 0.010695314486610617

## Discovered Archetypes

### Low Three Share Z Type (Archetype 0)

- **Players**: 52
- **Avg RAPM**: 0.62
- **Survival Rate**: 63.5%
- **Description**: Characterized by low Three Share Z
- **Examples**: Jimmy Butler III, Kyrie Irving, Tristan Thompson

### High Transfer Pace Delta Mean Type (Archetype 1)

- **Players**: 42
- **Avg RAPM**: 0.05
- **Survival Rate**: 66.7%
- **Description**: Characterized by high Transfer Pace Delta Mean
- **Examples**: Josh Selby, Nolan Smith, Edwin Ubiles

### Low Recruiting Rank Type (Archetype 2)

- **Players**: 39
- **Avg RAPM**: 0.40
- **Survival Rate**: 74.4%
- **Description**: Characterized by low Recruiting Rank
- **Examples**: Terrel Harris, Tobias Harris, Lance Thomas

### Low Recruiting Rating Type (Archetype 3)

- **Players**: 67
- **Avg RAPM**: 0.57
- **Survival Rate**: 67.2%
- **Description**: Characterized by low Recruiting Rating
- **Examples**: Ivan Johnson, Dennis Horner, Cory Higgins

### Low Recruiting Rank Type (Archetype 4)

- **Players**: 32
- **Avg RAPM**: 0.45
- **Survival Rate**: 71.9%
- **Description**: Characterized by low Recruiting Rank
- **Examples**: Kemba Walker, Shelvin Mack, Tony Wroten

### Low Shots Total Type (Archetype 5)

- **Players**: 31
- **Avg RAPM**: 0.23
- **Survival Rate**: 80.6%
- **Description**: Characterized by low Shots Total
- **Examples**: Kim English, Terrence Jones, Miles Plumlee

### Low Team Pace Type (Archetype 6)

- **Players**: 12
- **Avg RAPM**: 0.79
- **Survival Rate**: 66.7%
- **Description**: Characterized by low Team Pace
- **Examples**: Jordan Hamilton, Fab Melo, Jae Crowder

### High Recruiting Rating Type (Archetype 7)

- **Players**: 204
- **Avg RAPM**: 0.07
- **Survival Rate**: 73.0%
- **Description**: Characterized by high Recruiting Rating
- **Examples**: E'Twaun Moore, JaJuan Johnson, MarShon Brooks

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
