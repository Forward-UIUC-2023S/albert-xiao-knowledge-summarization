import torch
from datasets import Features, Sequence, Value, load_dataset, load_from_disk
import os
from transformers import (
    RagRetriever,
    RagSequenceForGeneration,
    RagTokenizer,
)

from flask import request
import urllib.parse

from flask import Flask

app = Flask(__name__)


import torch
from datasets import Features, Sequence, Value, load_dataset, load_from_disk
import os
from transformers import (
    RagRetriever,
    RagSequenceForGeneration,
    RagTokenizer,
)
from transformers import AutoTokenizer, RagRetriever, RagSequenceForGeneration, RagConfig, RagModel


_DIR_WIKI = '/scratch/ziyic2/wiki_playground/'
_DIR_WIKI = '/scratch/ziyic2/wiki_dpr_enterprise/'

passages_path = os.path.join(_DIR_WIKI, "enterprise_api_dense_retrieval_dataset")
index_path = os.path.join(_DIR_WIKI, "enterprise_api_dense_retrieval_dataset_hnsw_index.faiss")

true_dataset = load_from_disk(passages_path)
# true_dataset = load_from_disk(passages_path, keep_in_memory=False)
true_dataset.load_faiss_index('embeddings', index_path)
true_dataset.get_index("embeddings")

enterprise_retriever = RagRetriever.from_pretrained(
    'facebook/rag-sequence-nq', index_name="custom", indexed_dataset=true_dataset
)

retriever = RagRetriever.from_pretrained(
    'facebook/rag-sequence-nq', dataset="wiki_dpr", index_name="exact", use_dummy_dataset=True
)
rag_model = RagModel.from_pretrained("facebook/rag-sequence-nq", retriever=retriever)
rag_question_encoder = rag_model.question_encoder
# retriever_from_enterprise_api = RagSequenceForGeneration.from_pretrained('facebook/rag-sequence-nq', retriever=enterprise_retriever)
retriever_from_enterprise_api_tokenizer = RagTokenizer.from_pretrained('facebook/rag-sequence-nq')


def retrieve_dpr_enterprise_api(q, n_docs=5, question_encoder=rag_question_encoder):
    input_ids = retriever_from_enterprise_api_tokenizer(q, return_tensors="pt", padding=True)
    input_ids = input_ids['input_ids']
    embeddings = question_encoder(input_ids)
    res = enterprise_retriever(input_ids, embeddings[0].detach().to(torch.float32).numpy(), n_docs=n_docs)
    # context = []
    outputs_rag = []
    for context_input_ids in res.data['context_input_ids']:
        text = retriever_from_enterprise_api_tokenizer.decode(context_input_ids, skip_special_tokens=True)
        text = text[text.find(' / ') + 3:text.find(' // ')]
        outputs_rag.append(text)
    return '\n'.join(outputs_rag)


@app.route("/retreive_from_enterprise_api",methods = ['GET'])
def retreive_from_enterprise_api():
    query = request.args.get('query')
    k = request.args.get('k', 5)
    k = int(k)
    res = {'res': retrieve_dpr_enterprise_api(query, k)}
    return res

# @app.route("/retreive_from_wiki_dpr",methods = ['GET'])
# def retreive_from_wiki_dpr():
#     query = request.args.get('query')
#     k = request.args.get('k', 5)
#     k = int(k)
#     # query = request.form.get('query')
#     # k = request.form.get('k', 5)
#     res = {'res': retrieve_dpr_unmerged(query, k)}
#     return res

print("running")
app.run()