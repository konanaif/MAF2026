"""The complete public method/model registry; no optional hidden methods."""
MODELS = ('roberta-base', 'roberta-large', 'qwen3-4b')
METHODS = ('base', 'base_swa', 'fairbatch', 'leace_swa')
MODEL_IDS = {'qwen3-4b': 'Qwen/Qwen3-4B', 'roberta-base': 'klue/roberta-base', 'roberta-large': 'klue/roberta-large'}
MODEL_REVISIONS = {'qwen3-4b': '1cfa9a7208912126459214e8b04321603b3df60c',
                   'roberta-base': '02f94ba5e3fcb7e2a58a390b8639b0fac974a8da',
                   'roberta-large': '28d911204e9022eda172571ca8cc61eaffd942f7'}
