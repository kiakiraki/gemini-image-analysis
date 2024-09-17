import json
import os
import tempfile
import time

import google.generativeai as genai
import gradio as gr

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

PROMPT = """
あなたは動画分析アプリケーションです。
この動画のシーンを分析し、タイムスタンプ毎に描写されている内容を要約し100点満点で採点してください。
以下の要素がある場合は加点対象とします。
・人の声や歓声が聞こえる
・人物が正面を向いて写っている
・画面に動きがある
・人物の表情が豊かである
・イベントの動画である
・露出が適切である
・写っている人物が乳幼児・子どもである
以下の場合は減点対象とします。
・画面に動きがない
・人の声が入っていない
・人の顔が写っていない (大きく減点)
・画面がブレている (大きく減点)
・露出が不適切である (大きく減点)
・騒音が激しく、内容がわかりにくい
また、動画の中でベストなシーンを選定してください。

**出力は次のJSON形式で行ってください。**
```json
{
  "scenes": [
    {
      "timestamp": "タイムスタンプ（mm:ss）",
      "summary": "シーンの要約",
      "score": "採点結果（0〜100の整数）"
    },
    ...
  ],
  "best_scene": "ベストなシーンのタイムスタンプ（mm:ss）"
}
```

"""


def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file


def wait_for_files_active(files):
    """Waits for the given files to be active.

    Some files uploaded to the Gemini API need to be processed before they can be
    used as prompt inputs. The status can be seen by querying the file's "state"
    field.

    This implementation uses a simple blocking polling loop. Production code
    should probably employ a more sophisticated approach.
    """
    print("Waiting for file processing...")
    for name in (file.name for file in files):
        file = genai.get_file(name)
        while file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(10)
            file = genai.get_file(name)
        if file.state.name != "ACTIVE":
            raise Exception(f"File {file.name} failed to process")
    print("...all files ready")
    print()


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


def evaluate_video(video):
    """Evaluates the uploaded video using Gemini."""
    # 一時ファイルを作成
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        with open(video, "rb") as video_file:
            temp_file.write(video_file.read())
        temp_file_path = temp_file.name

    uploaded_file = upload_to_gemini(temp_file_path, mime_type="video/mp4")
    wait_for_files_active([uploaded_file])

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

    response = chat_session.send_message("この動画を分析してください")

    try:
        cleaned_response = response.text.strip()
        result = json.loads(cleaned_response)
    except json.JSONDecodeError:
        result = {"error": "Failed to parse response"}
    return result


# Create Gradio interface
iface = gr.Interface(
    fn=evaluate_video,
    inputs=gr.Video(label="アップロードする動画"),
    outputs="json",
    title="Video Analyzer",
    description="Analyze a video using the Gemini AI model.",
)

iface.launch()
