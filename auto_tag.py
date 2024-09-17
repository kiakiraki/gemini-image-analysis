import json
import os
import tempfile

import google.generativeai as genai
import gradio as gr
import pandas as pd
from PIL import Image

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

PROMPT = """
あなたは写真アルバムサービスに搭載されている自動タグ付け機能です。アップロードされた写真に対して、次の要件に従いタグを付与してください。
・タグは英語で付与する
・確信度の高い順に最大20個のタグを付与する
・確信度を有効数字3桁で出力する
・子どものイベントや人物に関するタグを重視する
・似たようなタグが多く出力されないようにする (例えば、wedding dress / wedding party などは wedding として1つのタグを付与する)
・似たような意味のタグは1つにまとめる (例えば、smile と laugh など)
・出力はJSON形式で、タグとその確信度の組み合わせをリストとして返す。
"""


def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini.

    See https://ai.google.dev/gemini-api/docs/prompting_with_media
    """
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file


# Create the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)


def evaluate_image(image):
    """Evaluates the uploaded image using Gemini."""
    image = Image.fromarray(image)

    # 一時ファイルを作成
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name, "PNG")
        temp_file_path = temp_file.name

    uploaded_file = upload_to_gemini(temp_file_path, mime_type="image/png")

    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [
                    uploaded_file,
                    PROMPT,
                ],
            },
        ]
    )

    response = chat_session.send_message("この写真にタグを付与してください")

    # JSON形式のレスポンスを解析し、タグと確信度のリストに変換
    try:
        tags_with_confidence = json.loads(response.text)
        df = pd.DataFrame(tags_with_confidence, columns=["tag", "confidence"])
    except json.JSONDecodeError as e:
        return f"Error parsing JSON response: {e}"

    return gr.DataFrame(value=df)  # DataFrameとして返す


# Create Gradio interface
iface = gr.Interface(
    fn=evaluate_image,
    inputs=gr.Image(label="アップロードする写真"),
    outputs=gr.DataFrame(headers=["Tag", "Confidence"], label="タグと確信度"),
    title="自動タグ付けアプリ",
)

iface.launch()
