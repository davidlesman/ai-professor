import json
from pathlib import Path
from django.http import JsonResponse
from django.shortcuts import render
from django.views import View
import google.generativeai as genai
import numpy as np
import pandas as pd

from rateprofessor import settings


with open(settings.BASE_DIR / "system_prompt.txt") as f:
    system_prompt = f.read()

genai.configure(api_key=settings.API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash", system_instruction=system_prompt)
chat = model.start_chat(history=[])


data = []
REVIEWS = Path(__file__).resolve().parent / "reviews.json"
with open(REVIEWS, "r") as f:
    for i, review in enumerate(json.load(f).get("reviews")):
        embedding = genai.embed_content(
            model="models/embedding-001", content=review, task_type="retrieval_query"
        )
        data.append({"text": review, "embedding": embedding["embedding"]})
        print(f"{i+1}/20")

# embedding df
df = pd.DataFrame(data)
print(df)


def find_best_passage(query, dataframe):
    """
    Compute the distances between the query and each document in the dataframe
    using the dot product.
    """
    query_embedding = genai.embed_content(
        model="models/embedding-001", content=query, task_type="retrieval_query"
    )
    dot_products = np.dot(
        np.stack(dataframe["embedding"]), query_embedding["embedding"]
    )
    idx = np.argmax(dot_products)
    return dataframe.iloc[idx]["text"]  # Return text from index with max value


class IndexPage(View):
    def get(self, request):

        return render(request, "web/index.html")


class Send(View):
    def post(self, request):
        data = json.loads(request.body)
        passage = find_best_passage(data.get("user_input"), df).replace("'", "").replace('"', "").replace("\n", " ")
        print(passage)
        response = chat.send_message(f"QUESTION: {data.get("user_input")} \n RELEVANT PASSAGE: {passage} \n ANSWER:")

        return JsonResponse({"content": response.text}, status=200)
