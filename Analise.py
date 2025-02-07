import boto3
import json
from PIL import Image, ImageDraw, ImageFont
from botocore.exceptions import ClientError
from typing import Dict, List

client_textract = boto3.client("textract")
client_rekognition = boto3.client("rekognition")


### Funções Comuns

def load_image_bytes(file_path: str) -> bytearray:
    try:
        with open(file_path, "rb") as file:
            return bytearray(file.read())
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado: {file_path}")
        raise


### OCR CNH Refatorado

def analyze_cnh(file_path: str) -> Dict:
    try:
        doc_bytes = load_image_bytes(file_path)
        response = client_textract.analyze_document(
            Document={"Bytes": doc_bytes}, FeatureTypes=["FORMS"]
        )
        return response
    except ClientError as e:
        print(f"Erro AWS Textract: {e}")
        return {}


### OCR Lista Escolar Refatorado

def detect_text(file_path: str) -> List[str]:
    try:
        doc_bytes = load_image_bytes(file_path)
        response = client_textract.detect_document_text(Document={"Bytes": doc_bytes})
        return [block["Text"] for block in response.get("Blocks", []) if block.get("BlockType") == "LINE"]
    except ClientError as e:
        print(f"Erro AWS Textract: {e}")
        return []


### Comparador de Rostos Refatorado

def compare_faces(source_image_path: str, target_image_path: str, similarity_threshold: int = 80) -> List:
    try:
        with open(source_image_path, "rb") as source_image, open(target_image_path, "rb") as target_image:
            response = client_rekognition.compare_faces(
                SourceImage={"Bytes": source_image.read()},
                TargetImage={"Bytes": target_image.read()},
                SimilarityThreshold=similarity_threshold,
            )
        return response.get("FaceMatches", [])
    except ClientError as e:
        print(f"Erro AWS Rekognition: {e}")
        return []


### Reconhecimento de Celebridades Refatorado

def recognize_celebrities(photo_path: str) -> List[Dict]:
    try:
        with open(photo_path, "rb") as image:
            response = client_rekognition.recognize_celebrities(Image={"Bytes": image.read()})
        return response.get("CelebrityFaces", [])
    except ClientError as e:
        print(f"Erro AWS Rekognition: {e}")
        return []


def draw_boxes(image_path: str, output_path: str, faces: List[Dict]):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("Ubuntu-R.ttf", 20)
    width, height = image.size

    for face in faces:
        box = face["Face"]["BoundingBox"]
        left = int(box["Left"] * width)
        top = int(box["Top"] * height)
        right = int((box["Left"] + box["Width"]) * width)
        bottom = int((box["Top"] + box["Height"]) * height)

        draw.rectangle([left, top, right, bottom], outline="red", width=3)
        name = face.get("Name", "")
        draw.text((left, top - 20), name, font=font, fill="white")

    image.save(output_path)
    print(f"Imagem salva: {output_path}")


### Exemplo de Uso
if __name__ == "__main__":
    cnh_file = "./images/cnh.png"
    print("=== Dados da CNH ===")
    cnh_response = analyze_cnh(cnh_file)
    print(json.dumps(cnh_response, indent=2))

    lista_escolar_file = "./images/lista-material-escolar.jpeg"
    print("=== Linhas detectadas na lista ===")
    for line in detect_text(lista_escolar_file):
        print(line)

    print("=== Comparando Rostos ===")
    source_image = "./images/neymar.jpg"
    target_image = "./images/msn.jpg"
    face_matches = compare_faces(source_image, target_image)
    for match in face_matches:
        print(f"Similaridade: {match["Similarity"]}%")

    print("=== Reconhecendo Celebridades ===")
    celeb_image = "./images/neymar-torcedores.jpg"
    celeb_faces = recognize_celebrities(celeb_image)
    draw_boxes(celeb_image, "./images/resultado_celebridades.jpg", celeb_faces)
