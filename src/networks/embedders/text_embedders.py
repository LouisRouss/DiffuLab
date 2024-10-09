import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoTokenizer, CLIPModel

from networks.common import contextEmbedder

class text_MMDiTEmbedder(contextEmbedder)