

import logging


logging.basicConfig(
    format='%(asctime)s | %(levelname)s : %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mlm-var.log'),
    ]
)

logger = logging.getLogger('mlm-var')
