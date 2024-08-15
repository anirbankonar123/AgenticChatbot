# import google.generativeai as genai
from PIL import Image
import os
import base64
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
# img = Image.open('img/page_15.png')
#
# model = genai.GenerativeModel(model_name="gemini-1.5-flash")
# response = model.generate_content(["What is the mass of moon?", img])
# print(response.text)

# Initialize the model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
# Convert the image to base64 format
query = "What is the night side atmospheric pressure on Moon?"
def get_imgdata(file_path):
    with open(file_path, "rb") as f:
        encoded_image = base64.b64encode(f.read())
    image_data = encoded_image.decode('utf-8')
    return image_data

base_dir = "data_output"
image_data_arr=[]

#To test on image array
# base_dir = "data_output"
# for file in os.listdir(base_dir):
#     image_data = get_imgdata(os.path.join(base_dir,file))
#     image_data_arr.append(image_data)
#
#
# content=[]
# data = {}
# data["type"] = "text"
# data["text"] = query
# content.append(data)
#
# for image_data in image_data_arr:
#     img = {}
#     img_data = {}
#     img["type"] = "image_url"
#
#     img_data ["url"]=f"data:image/jpeg;base64,{image_data}"
#     img["image_url"] = img_data
#     content.append(img)
#
# message = HumanMessage(content)
#
# response = model.invoke([message])
#
# print(response.content)

#To test on one image
image_data = get_imgdata("data_output/page_15.png")

message = HumanMessage(
    content=[
        {"type": "text", "text": query},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
        },

    ],
)

response = model.invoke([message])
print(response.content)


