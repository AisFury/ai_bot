import gradio as gr

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
import torch
from threading import Thread
from sentence_transformers import SentenceTransformer

ST = SentenceTransformer("DeepPavlov/rubert-base-cased", device="cuda")

def get_similarity(query: str, model: SentenceTransformer, database_emb, base_texts):

    query_emb = model.encode(query, convert_to_tensor=True, device="cuda")

    database_emb = torch.from_numpy(database_emb).to("cuda")

    similaritires = util.pytorch_cos_sim(query_emb, database_emb)

    reults = pd.DataFrame(base_texts.values, columns=["texts"])

    reults["cos_sim"] = similaritires.cpu().reshape(-1,1)

    return reults

databes_text = pd.read_excel("База знаний.xlsx")["example"]
database_emb = np.vstack(np.load("base knowledge embs.npy", allow_pickle=True)).astype(np.float32)


model_id = r"C:\Users\ilyam\Desktop\schoolDS_progect/Llama-3.2-3B-Instruct/"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=bnb_config,
)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

SYS_PROMPT = """Ты являешься помощником в ответах на вопросы.
Тебе выдают фрагменты длинного документа и вопрос. Отвечай на вопросы в разговорной форме.
Если ты не знаешь ответа, просто скажи "Я не знаю". Не придумывай ответ."""



def search(query: str, k: int = 4):
    sim_reslut = get_similarity(query, ST, database_emb=database_emb, base_texts=databes_text)
    sim_reslut = sim_reslut.sort_values(by="cos_sim", ascending=False).head(k)

    return sim_reslut

def format_prompt(prompt,retrieved_documents,k):
    PROMPT = f"Question:{prompt}\nContext:"

    if retrieved_documents["cos_sim"].iloc[0] < 0.45:
        PROMPT+= f"В твой базе знаний нет ответа на данный вопрос. Тебе нужно ответить 'Я не знаю'."
        return PROMPT

    for idx in range(k) :
        PROMPT+= f"Вот что есть в базе зназний:\n {retrieved_documents["texts"].iloc[idx]}\n"
    return PROMPT


def talk(prompt, history):

    top = 4

    retrieved_documents = search(prompt, top)

    formatted_prompt = format_prompt(prompt,retrieved_documents,top)

    messages = [{"role":"system","content":SYS_PROMPT},
                {"role":"user","content":formatted_prompt}]


    input_ids = tokenizer.apply_chat_template(
      messages,
      add_generation_prompt=True,
      return_tensors="pt"
    ).to(model.device)
    outputs = model.generate(
      input_ids,
      max_new_tokens=1024,
      eos_token_id=terminators,
      do_sample=True,
      temperature=0.6,
      top_p=0.9,
    )
    streamer = TextIteratorStreamer(
            tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
        )
    generate_kwargs = dict(
        input_ids= input_ids,
        streamer=streamer,
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.95,
        temperature=0.60,
        eos_token_id=terminators,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        print(outputs)
        yield "".join(outputs)


demo = gr.ChatInterface(
    fn=talk,
    chatbot=gr.Chatbot(
        show_label=True,
        show_share_button=True,
        show_copy_button=True,
        layout="bubble",
        bubble_full_width=False,
    ),
    examples=[["Что такое Process Mining?"]]
    
)
demo.launch(debug=True)
