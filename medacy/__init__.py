__version__ = '0.1.1'
__authors__ = "Andriy Mulyar, Jorge Vargas, Corey Sutphin, Bobby Best, Steele Farnsworth, Bridget T. McInnes"


# These classes are part of the interface, therefore they're available to import directly from the top-level
# Do not import `from medacy import Model` internally.
from medacy.data.dataset import Dataset
from medacy.model.model import Model
from medacy.pipeline_components.feature_overlayers.metamap.metamap import MetaMap
