import os
import base64
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI


class ImageExtractor(object):

    def __init__(self):
        global model
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    def get_imgdata(self,file_path):
        with open(file_path, "rb") as f:
            encoded_image = base64.b64encode(f.read())
        image_data = encoded_image.decode('utf-8')
        return image_data

    def get_response(self,query,base_dir):
        image_data_arr = []

        for file in os.listdir(base_dir):
            image_data = self.get_imgdata(os.path.join(base_dir, file))
            image_data_arr.append(image_data)

        content = []
        data = {}
        data["type"] = "text"
        data["text"] = query
        content.append(data)

        for image_data in image_data_arr:
            img = {}
            img_data = {}
            img["type"] = "image_url"

            img_data["url"] = f"data:image/jpeg;base64,{image_data}"
            img["image_url"] = img_data
            content.append(img)

        message = HumanMessage(content)

        response = model.invoke([message])

        print(response.content)
        return str(response.content)