# Latent Space Model Analysis Report

**Generated**: 2026-02-18 20:26
**Model**: latent_model_20260218_202635

## Training Results

- **Best Epoch**: 6
- **Best Val Loss**: 1.6895
- **Test RMSE**: 2.000478744506836
- **Test Correlation**: -0.41301038175653376

## Discovered Archetypes

### Low Dev Quality Weight Type (Archetype 0)

- **Players**: 159
- **Avg RAPM**: 0.59
- **Survival Rate**: 81.1%
- **Description**: Characterized by low Dev Quality Weight
- **Examples**: Ivan Johnson, Jimmy Butler III, Cory Higgins

### High Recruiting Rank Type (Archetype 1)

- **Players**: 76
- **Avg RAPM**: 0.19
- **Survival Rate**: 69.7%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: MarShon Brooks, D.J. Kennedy, Derrick Williams

### High Transfer Max Shock Type (Archetype 2)

- **Players**: 72
- **Avg RAPM**: -0.16
- **Survival Rate**: 59.7%
- **Description**: Characterized by high Transfer Max Shock
- **Examples**: Jordan Clarkson, Rodney Hood, Dorian Finney-Smith

### High Recruiting Rating Type (Archetype 3)

- **Players**: 117
- **Avg RAPM**: 0.12
- **Survival Rate**: 70.1%
- **Description**: Characterized by high Recruiting Rating
- **Examples**: Jordan Williams, Klay Thompson, Marcus Morris Sr.

### High Recruiting Rank Type (Archetype 4)

- **Players**: 49
- **Avg RAPM**: 0.20
- **Survival Rate**: 75.5%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Kenneth Faried, Norris Cole, Justin Harper

### High Avg Shot Dist Type (Archetype 5)

- **Players**: 52
- **Avg RAPM**: -0.18
- **Survival Rate**: 53.8%
- **Description**: Characterized by high Avg Shot Dist
- **Examples**: Tacko Fall, Jeremiah Martin, Cody Martin

### Low Recruiting Rating Type (Archetype 6)

- **Players**: 204
- **Avg RAPM**: 0.14
- **Survival Rate**: 61.8%
- **Description**: Characterized by low Recruiting Rating
- **Examples**: E'Twaun Moore, JaJuan Johnson, Kemba Walker

### High Xy Rim Shots Type (Archetype 7)

- **Players**: 88
- **Avg RAPM**: 0.08
- **Survival Rate**: 68.2%
- **Description**: Characterized by high Xy Rim Shots
- **Examples**: Jeff Withey, Charlie Brown Jr., Romeo Langford

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
