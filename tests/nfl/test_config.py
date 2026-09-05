from sportsmodel.nfl.config import load_rating, load_gameline
from sportsmodel.nfl.elo import EloConfig
from sportsmodel.nfl.ratings import BlendConfig
from sportsmodel.nfl.gameline import GameLineConfig

def test_load_rating():
    elo, blend = load_rating()
    assert isinstance(elo, EloConfig) and isinstance(blend, BlendConfig)
    assert elo.k > 0 and 0 <= blend.w_sos <= 1

def test_load_gameline_uses_fitted_sigma_not_default():
    gl = load_gameline()
    assert isinstance(gl, GameLineConfig)
    assert gl.sigma_total != 10.0        # fitted (~13.58), not the illustrative default
    assert gl.offset == 75
