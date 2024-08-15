import fitz
from PIL import Image, ImageDraw
output_directory_path = "data_output2"
pdf_file = "data/An_Introduction_to_Space_Exploration_TPDas.pdf"
    #"data/The_Lunar_Saga_edition2.pdf"
import torch

pdf_document = fitz.open(pdf_file)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Iterate through each page and convert to an image
for page_number in range(pdf_document.page_count):
    # Get the page
    page = pdf_document[page_number]

    # Convert the page to an image
    pix = page.get_pixmap()

    # Create a Pillow Image object from the pixmap
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Save the image
    image.save(f"./{output_directory_path}/page_{page_number + 1}.png")

# Close the PDF file
pdf_document.close()