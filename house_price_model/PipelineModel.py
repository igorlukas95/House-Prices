from house_price_model.Config.core import _config
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso


model_pipeline = Pipeline([
    ('lasso', Lasso(
        alpha=_config.config_model.alpha,
        random_state=_config.config_model.random_state
    ))
])



