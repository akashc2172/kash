from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from typing import Tuple, Optional

def compute_rapm_from_stints(stints: pd.DataFrame, lambda_: float=2000.0) -> pd.DataFrame:
    '''
    stints: rows with teamId, gameId, lineup athletes list, possessions, net_points_per100 (or margin per 100).
    This is a simplified ridge APM implementation.
    '''
    # Expect columns: athleteIds (list), possessions, netRating (per100)
    df = stints.copy()
    # parse athleteIds column
    def parse_ids(x):
        if isinstance(x, list): return x
        if isinstance(x, str):
            try:
                import json
                v=json.loads(x)
                if isinstance(v,list): return v
            except Exception:
                return []
        return []
    if "athleteIds" in df.columns:
        df["_ath"]=df["athleteIds"].apply(parse_ids)
    else:
        return pd.DataFrame()

    df["possessions"]=pd.to_numeric(df.get("teamStats.possessions", df.get("possessions", 0)), errors="coerce").fillna(0.0)
    y = pd.to_numeric(df.get("netRating", np.nan), errors="coerce")
    # If netRating isn't present, cannot.
    if y.isna().all():
        return pd.DataFrame()
    y = y.fillna(0.0).values  # per100
    w = df["possessions"].values
    # Create player index
    players = sorted({aid for lst in df["_ath"] for aid in lst})
    p2i = {p:i for i,p in enumerate(players)}
    X = np.zeros((len(df), len(players)), dtype=float)
    for r, lst in enumerate(df["_ath"]):
        for aid in lst:
            X[r, p2i[aid]] = 1.0
    # Weight by possessions (sqrt)
    sw = np.sqrt(np.maximum(w, 0.0))
    Xw = X * sw[:,None]
    yw = y * sw
    ridge = Ridge(alpha=lambda_, fit_intercept=True)
    ridge.fit(Xw, yw)
    coef = ridge.coef_
    out = pd.DataFrame({"athleteId": players, "rapm_total": coef})
    return out
