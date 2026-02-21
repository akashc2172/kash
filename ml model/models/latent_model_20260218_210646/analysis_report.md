# Latent Space Model Analysis Report

**Generated**: 2026-02-18 21:06
**Model**: latent_model_20260218_210646

## Training Results

- **Best Epoch**: 33
- **Best Val Loss**: 1.5709
- **Test RMSE**: 1.6479166746139526
- **Test Correlation**: -0.03290323441109302

## Discovered Archetypes

### High Recruiting Rank Type (Archetype 0)

- **Players**: 48
- **Avg RAPM**: 0.42
- **Survival Rate**: 77.1%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Edmond Sumner, Jamel Artis, John Collins

### High Blk Total Per100Poss Type (Archetype 1)

- **Players**: 56
- **Avg RAPM**: 1.09
- **Survival Rate**: 66.1%
- **Description**: Characterized by high Blk Total Per100Poss
- **Examples**: Mangok Mathiang, Amile Jefferson, Bam Adebayo

### High Ast Total Per100Poss Type (Archetype 2)

- **Players**: 154
- **Avg RAPM**: -0.03
- **Survival Rate**: 70.8%
- **Description**: Characterized by high Ast Total Per100Poss
- **Examples**: Erik McCree, Jaylen Morris, PJ Dozier

### High Blk Total Per100Poss Type (Archetype 3)

- **Players**: 16
- **Avg RAPM**: 0.41
- **Survival Rate**: 75.0%
- **Description**: Characterized by high Blk Total Per100Poss
- **Examples**: Eric Moreland, Darius Miller, Terrence Jones

### Low Recruiting Rating Type (Archetype 4)

- **Players**: 47
- **Avg RAPM**: 0.21
- **Survival Rate**: 74.5%
- **Description**: Characterized by low Recruiting Rating
- **Examples**: Tyler Cavanaugh, Charles Cooke, Jayson Tatum

### Low Recruiting Rating Type (Archetype 5)

- **Players**: 38
- **Avg RAPM**: 0.95
- **Survival Rate**: 71.1%
- **Description**: Characterized by low Recruiting Rating
- **Examples**: Jonathan Isaac, Jaylen Brown, Jake Layman

### High Recruiting Rating Type (Archetype 6)

- **Players**: 78
- **Avg RAPM**: -0.09
- **Survival Rate**: 70.5%
- **Description**: Characterized by high Recruiting Rating
- **Examples**: Josh Hart, London Perrantes, Nigel Williams-Goss

### High Recruiting Rank Type (Archetype 7)

- **Players**: 46
- **Avg RAPM**: 0.06
- **Survival Rate**: 67.4%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Ike Anigbogu, Luke Kornet, Ben Moore

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
