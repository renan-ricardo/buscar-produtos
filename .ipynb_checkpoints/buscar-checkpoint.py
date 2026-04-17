# buscar.py
import sys, os, json, requests, io, torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

foto_url   = sys.argv[1]
quantidade = int(sys.argv[2]) if len(sys.argv) > 2 else 5

# Carrega modelo
model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
sb        = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Baixa e vetoriza a foto
response = requests.get(foto_url)
image    = Image.open(io.BytesIO(response.content)).convert("RGB")
inputs   = processor(images=image, return_tensors="pt")

with torch.no_grad():
    embedding = model.get_image_features(**inputs)
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    vetor     = embedding[0].tolist()

# Busca no Supabase
result = sb.rpc("buscar_produto_similar", {
    "query_embedding": vetor,
    "match_count":     quantidade
}).execute()

# Retorna JSON para o N8N capturar
print(json.dumps({"resultados": result.data}))
