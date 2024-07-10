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
pdf_path = "C:\\Users\\a813538\\Downloads\\rouani_azedine_cv.pdf"

# Extraire le texte du fichier PDF - B
pdf_text = extract_text_from_pdf(pdf_path)

# Récupérer l'input user
user_input = input("What is your question about this profile? ")

# Appel à Ollama
response = ollama.chat(model='gemma', messages=[
  {
    'role': 'user',
    'content': user_input + ' of this candidate : ' + pdf_text
  },
])
print(response['message']['content'])