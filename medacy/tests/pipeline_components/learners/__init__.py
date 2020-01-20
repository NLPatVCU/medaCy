from medacy.tools.read_config import read_config

cuda_device = read_config('cuda_device')
use_cuda = cuda_device >= -1

word_embeddings = read_config('word_embeddings')