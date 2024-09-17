import json
import os
import tempfile

import google.generativeai as genai
import gradio as gr
from PIL import Image

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

PROMPT = """
あなたは写真を採点するプログラムです。次の基準を用いて、アップロードされた写真を100点満点で採点してください。

加点要素:
・人物の顔がはっきりと適切な露出で写っている
・人物の豊かな表情が写っている
・誕生日やパーティー、季節の行事など、イベントの写真である
減点要素:
・人物の顔が写っていない場合は大きく減点
・人物の顔が無表情である
・顔が見切れている
・ブレ、ピンボケ、露出のズレなど、撮影に失敗した写真である

**採点結果は必ず次のJSON形式で出力してください。**
```json
{
    "score": 採点結果（0〜100の整数）,
    "reason": 採点理由",
```
"""


def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
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
    with tempfile.NamedTemporaryFile(delete=True, suffix=".png") as temp_file:
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

    response = chat_session.send_message("次の写真を採点してください。")

    try:
        cleaned_response = response.text.strip()
        result = json.loads(cleaned_response)
        score = result["score"]
        reason = result["reason"]
        return score, reason
    except json.JSONDecodeError:
        return "エラー: JSON形式のレスポンスが得られませんでした。", ""


# Create Gradio interface
iface = gr.Interface(
    fn=evaluate_image,
    inputs=gr.Image(label="アップロードする写真"),
    outputs=[gr.Textbox(label="点数"), gr.Textbox(label="採点理由")],
    title="写真採点アプリ",
)

iface.launch()
