from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

# Model names for embedding generation
tokenizer_name = 'microsoft/codebert-base'
embedding_model_name = 'microsoft/codebert-base'
cross_encoder_tokenizer_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
cross_encoder_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

# RAG model for response generation
agent_model_name = 'llama3.2'

# Model names for repository parsing
code_parser_model_name = 'codellama'
readme_generator_model_name = 'codellama'

# Load model for embedding generation
embedding_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
embedding_model = AutoModel.from_pretrained(embedding_model_name)

# Load cross-encoder model for reranking
cross_encoder_tokenizer = AutoTokenizer.from_pretrained(cross_encoder_tokenizer_name)
cross_encoder_model = AutoModelForSequenceClassification.from_pretrained(cross_encoder_name)
