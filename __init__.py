__version__ = '0.1.0'

# Import the single function from each module
from .preprocessing import preprocessing
from .tool import tool
from .visualisation import visualisation

# Expose these functions as part of the package
__all__ = ['preprocessing', 'tool', 'visualisation']
