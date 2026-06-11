"""Feature engineering — WoE/IV + encoders + scalers."""

from rsm_pipeline.feature_engineering.apply import (
    apply_feature_engineering,
)  # noqa: F401
from rsm_pipeline.feature_engineering.factory import (  # noqa: F401
    ResolvedFESpec,
    build_feature_engineer,
)
from rsm_pipeline.feature_engineering.schema import (  # noqa: F401
    CategoricalFEDefaults,
    ColumnFEOverride,
    FeatureEngineeringConfig,
    NumericFEDefaults,
)
from rsm_pipeline.feature_engineering.woe.iv import (  # noqa: F401
    WoEBin,
    WoETable,
    compute_iv,
)
