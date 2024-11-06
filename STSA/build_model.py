
from STSA.models.STSA import STSA

def build_model(args):
    model = STSA(dataset=args.dataset)
    return model
