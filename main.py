import gradio as gr

import torch
from transformers import pipeline
 
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(device)

def ask(text):
  model_id = r"C:\Users\ilyam\Desktop\schoolDS_progect\Llama-3.2-3B-Instruct"

  pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device = device, 
  )

  messages = [
    {"role": "system", "content": "Ты чатбот помогающий отвечать на вопросы."},
    {"role": "user", "content": text},
  ] 

  outputs = pipe(
      messages,
      max_new_tokens=256,
  )
  return outputs[0]["generated_text"][-1]["content"] 
with gr.Blocks() as server:
  with gr.Tab("LLM Inferencing"):
 
    model_input = gr.Textbox(label="Ваш вопрос:", 
                             value="Какой ваш вопрос?", interactive=True)
    ask_button = gr.Button("Ask")
    model_output = gr.Textbox(label="Ответ:", interactive=False, 
                              value="Ответ вам..")
 
  ask_button.click(ask, inputs=[model_input], outputs=[model_output])

server.launch()