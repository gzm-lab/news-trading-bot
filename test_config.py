import sys
import os
sys.path.insert(0, os.path.abspath('.'))
from src.config import settings
print('Strategy dict:', settings.strategy.model_dump())
print('Buy threshold:', getattr(settings.strategy, 'buy_threshold', 'MISSING'))

