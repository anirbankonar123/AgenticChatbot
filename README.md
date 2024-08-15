<h1> Streamlit chatbot with Agentic RAG and Information Extractor </h1>

<p>To run the chatbot, install the necessary libs from requirements.txt <br>
The solution uses OpenAI, Mistral AI with llamaindex for the Agentic RAG and Google Gemini-1.5-Flash for Image text inference<br>
Signup for Openai, Mistral AI, and Google Gemini (Generative AI Studio), and get the keys for each.<br>
Set the env variables:<br>
export OPENAI_API_KEY=<br>
export MISTRAL_API_KEY=<br>
export GOOGLE_API_KEY=<br>

Put your PDF files in a folder "data"<br>

Run the multiExtractImages.py, by giving the PDF file path and output dir for each pdf. This will extract
the images from the PDF and store in output folders<br>

Edit the chatbot.py and make sure the same image output folders are specified in line 149,150,151
msg = img_ext.get_response(query,"data_output") <br>

Make sure the same data folder is specified in line 12
agent=multiDocAgent.get_agent("data")<br>

Run the chatbot using : streamlit run chatbot.py<br>
</p>

<p> To test the Agentic RAG:<br>
Use cmd : python testAgent.py<br>
Specify the correct data folder, and input a query<br>
</p>

<p> To test the Image based RAG:<br>
Use cmd : python test_ImgExt.py<br>
Specify the correct image data folder, and input a query<br>
</p>
