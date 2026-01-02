from .analytical_crps import (
    crps_analytical,
    crps_analytical_normal,
    crps_analytical_studentt,
)
from .ensemble_crps import crps_ensemble, crps_ensemble_naive
from .integral_crps import crps_integral
from .normalization import normalize_by_observation

crps_analytical_normalized = normalize_by_observation(crps_analytical)
crps_analytical_normal_normalized = normalize_by_observation(crps_analytical_normal)
crps_analytical_studentt_normalized = normalize_by_observation(crps_analytical_studentt)
crps_ensemble_normalized = normalize_by_observation(crps_ensemble)
crps_integral_normalized = normalize_by_observation(crps_integral)

__all__ = [
    "crps_analytical",
    "crps_analytical_normal",
    "crps_analytical_normal_normalized",
    "crps_analytical_normalized",
    "crps_analytical_studentt",
    "crps_analytical_studentt_normalized",
    "crps_ensemble",
    "crps_ensemble_naive",
    "crps_ensemble_normalized",
    "crps_integral",
    "crps_integral_normalized",
]
