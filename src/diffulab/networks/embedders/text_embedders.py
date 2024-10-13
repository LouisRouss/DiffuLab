import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoTokenizer, CLIPModel

from diffulab.networks.common import contextEmbedder

class text_MMDiTEmbedder(contextEmbedder)