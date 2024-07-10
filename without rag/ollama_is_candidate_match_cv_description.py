import ollama
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    # Ouvrir le document PDF
    doc = fitz.open(pdf_path)
    text = ""
    # Parcourir chaque page du document PDF
    for page in doc:
        text += page.get_text()
    return text

# Spécifiez le chemin vers votre fichier PDF - A
pdf_pathA = "C:\\Users\\a813538\\Downloads\\php_cv.pdf"
# Spécifiez le chemin vers votre fichier PDF - B
pdf_pathB = "C:\\Users\\a813538\\Downloads\\description_job.pdf"

# Extraire le texte du fichier PDF - B
pdf_textA = extract_text_from_pdf(pdf_pathA)
pdf_textB = extract_text_from_pdf(pdf_pathB)

# Récupérer du contenu avec le contexte
content = " is this candidate : " + pdf_textA + "a good match for this job : " + pdf_textB

# Choisir un modèle parmi les modèles installés 
model = 'gemma2'
# Appel à Ollama
response = ollama.chat(model=model, messages=[
  {
    'role': 'user',
    'content': content
  },
])
print(response['message']['content'])