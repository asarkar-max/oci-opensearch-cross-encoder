# Custom Cross-Encoder Deployment on OCI OpenSearch

**Context:** Oracle Cloud (OCI) Search with OpenSearch uses a heavily curated list of pre-trained models. They currently restrict this list mostly to dense embedding models (like standard `sentence-transformers`), leaving cross-encoders off the built-in list for now.

To bypass this and deploy a Hugging Face cross-encoder (`ms-marco-MiniLM-L-12-v2`), follow the steps below.

## Step 1: Trace and Package the Model
OpenSearch requires custom models to be serialized into a TorchScript (`.pt`) file and zipped alongside their `tokenizer.json` file. 

Save the script below as a Python file and run it locally. It will generate a `ms-marco-MiniLM-L-12-v2-fixed.zip` file. *(Note: This explicitly passes `token_type_ids` to ensure compatibility with OpenSearch's DJL engine).*

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import zipfile

model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# 1. Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, torchscript=True)

# 2. Create dummy inputs
dummy_text = ["query text", "document text"]
encoded = tokenizer(dummy_text[0], dummy_text[1], return_tensors='pt', max_length=512, padding='max_length', truncation=True)

# 3. Trace the model (THE FIX: Added token_type_ids)
traced_model = torch.jit.trace(model, (encoded['input_ids'], encoded['attention_mask'], encoded['token_type_ids']))
torch.jit.save(traced_model, "model.pt")

# 4. Save tokenizer.json
tokenizer.save_pretrained('./tokenizer_files')

# 5. Package into the required OpenSearch format
with zipfile.ZipFile('ms-marco-MiniLM-L-12-v2-fixed.zip', 'w') as zipf:
    zipf.write('model.pt')
    zipf.write('./tokenizer_files/tokenizer.json', arcname='tokenizer.json')

print("Success! Your fixed ms-marco-MiniLM-L-12-v2-fixed.zip is ready.")
```

## Step 2: Host the Model
Upload ms-marco-MiniLM-L-12-v2-fixed.zip to an OCI Object Storage bucket.

Create a Pre-Authenticated Request (PAR) for the file. Note this <PAR_URL>.

## Step 3: Configure and Deploy via OpenSearch Dev Tools
1. Enable ML on Data Nodes
<pre>
PUT /_cluster/settings
{
  "persistent": {
    "plugins.ml_commons.only_run_on_ml_node": "false",
    "plugins.ml_commons.model_access_control_enabled": "true",
    "plugins.ml_commons.native_memory_threshold": "99"
  }
}
</pre>

2. Register the ML Group
Take note of the `model_group_id` returned by this command.
<pre>
POST /_plugins/_ml/model_groups/_register
{
  "name": "cross_encoder_rerankers",
  "description": "Group for Hugging Face cross-encoders zero cost"
}
</pre>

3. Calculate the File Checksum
Run this locally against your zip file to get the `<FILE_HASH>`.
<pre>
shasum -a 256 ms-marco-MiniLM-L-12-v2-fixed.zip
</pre>

4. Register the Custom Model
This returns a `task_id`. Take note of it.

<pre>
POST /_plugins/_ml/models/_register
{
  "name": "custom-ms-marco-minilm-l12",
  "version": "1.0.0",
  "description": "Custom uploaded cross-encoder",
  "model_format": "TORCH_SCRIPT",
  "function_name": "TEXT_SIMILARITY",
  "model_group_id": "[GROUP_ID]",
  "model_content_hash_value": "[FILE_HASH]",
  "model_config": {
    "model_type": "cross_encoder",
    "embedding_dimension": 1
  },
  "url": "[PAR_URL]",
}
</pre>

5. Verify Registration
Run this until the state shows "COMPLETED". The response will contain the new `<model_id>`. Note it down.
<pre>
GET /_plugins/_ml/tasks/[TASK_ID]
</pre>

6. Deploy the Model
This also returns a task_id.
<pre>
POST /_plugins/_ml/models/[MODEL_ID]/_deploy
</pre>

## Step 4: Testing and Pipeline Integration
1. Test the Model Directly Verify the model is scoring documents correctly using mock data.
<pre>
POST /_plugins/_ml/models/[MODEL_ID]/_predict
{
  "query_text": "Holidays 2026",
  "text_docs": [
    "U.S. holidays 2026 list",
    "The festive season starts now...",
    "P.T.O and maternity leave",
    "New updated cloud infra from ",
    "System is on maintenance",
    "receives military friendly award"
  ]
}
</pre>

2. Deploy the Search Pipeline
Wire the model to evaluate the title and body_plain fields of incoming search results. (It should return "acknowledged": true).
<pre>
PUT /_search/pipeline/cross_encoder_pipeline
{
  "description": "Pipeline for MiniLM L12 cross-encoder reranking",
  "response_processors": [
    {
      "rerank": {
        "ml_opensearch": {
          "model_id": "[model_id]"
        },
        "context": {
          "document_fields": ["title", "body_plain"]
        }
      }
    }
  ]
}
</pre>

3. Execute a Reranked Search
Append the pipeline to your standard search query to see the reranked results.
<pre>
GET /[your_actual_index_name]/_search?search_pipeline=cross_encoder_pipeline
{
  "_source": ["title", "body_plain", "url"],
  "size": 10,
  "query": {
    "multi_match": {
      "query": "U.S. Holidays",
      "fields": ["title", "body_plain"]
    }
  },
  "ext": {
    "rerank": {
      "query_context": {
         "query_text": "2026 U.S. Holidays"
      }
    }
  }
}
</pre>
