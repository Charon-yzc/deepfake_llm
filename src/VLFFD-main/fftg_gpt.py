import base64
import requests
import os
import json
import pdb
import re
import random
from collections import defaultdict
import glob
import time
from tqdm import tqdm
from google import genai
from google.genai import types

def read_deepfake_file(file_path, n_frames=5):
    video_frames = defaultdict(list)
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('*')
            if len(parts) != 2:
                continue
            
            path, description = parts
            video_id = path.split('/')[-2]  # 获取视频ID (如 '000_003')
            
            video_frames[video_id].append((path, description))
    
    selected_frames = []
    
    for video_id, frames in video_frames.items():
        # 如果帧数少于n，全部选择
        if len(frames) <= n_frames:
            selected_frames.extend(frames)
        else:
            # 随机选择n帧
            selected = random.sample(frames, n_frames)
            selected_frames.extend(selected)
    
    return selected_frames
def get_oppopath(original_path):
    video_id = original_path.split('/')[-2]  # 获取视频ID (如 '000_003')
    opposite_path = original_path.replace("manipulated_sequences","original_sequences")
    opposite_path = opposite_path.split("/")
    opposite_path[-2] = video_id.split("_")[0]
    opposite_path[-5] = "youtube"
    opposite_path = "/".join(opposite_path)
    if not os.path.exists(opposite_path):
        original_folder = '/'.join(opposite_path.split('/')[:-1])+'/'
        # pdb.set_trace()
        face_paths = glob.glob(os.path.join(original_folder, '*.png'))
        opposite_path = random.choice(face_paths)
    return opposite_path


def load_existing_results(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_results(results, file_path):
    # 步骤1：读取现有数据（如果文件存在）
    existing_data = []
    if os.path.exists(results_file) and os.path.getsize(results_file) > 0:
        with open(results_file, "r") as f:
            existing_data = json.load(f)

    # 步骤2：追加新数据并写入
    existing_data+=results
    with open(results_file, "w") as f:
        json.dump(existing_data, f, indent=2)



# os.environ["all_proxy"] = ""
# # OpenAI API Key
# api_key = ""
# headers = {
#     "Content-Type": "application/json",
#     "Authorization": f"Bearer {api_key}"
#     }
# # Function to encode the image
# def encode_image(image_path):
#   with open(image_path, "rb") as image_file:
#     return base64.b64encode(image_file.read()).decode('utf-8')
  
# method = "Deepfakes"

method = "Face2Face"
# method = "FaceSwap"
# method = "NeuralTextures"


file_path = f'./output/{method}_descriptions.txt'

n_frames = 3  # 每个视频选择的帧数
results_file = f'./output/{method}_analysis_results_mini.json'

# 加载已有的结果
results = load_existing_results(results_file)
# 创建一个集合来存储已处理的视频ID
processed_video_ids = set()
for data in results:
    if data['answer']=="fake":
        video_id = data['imagepath'].split('/')[-2]
        processed_video_ids.add(video_id)

selected_frames = read_deepfake_file(file_path, n_frames)

# 打印结果
# for path, description in selected_frames:
for path, description in tqdm(selected_frames, desc="Processing videos"):
    video_id = path.split('/')[-2]
    if video_id in processed_video_ids:
        print(f"Skipping already processed video: {video_id}")
        continue
    
    # base64_image_fake = encode_image(path)
    real_path = get_oppopath(path)
    # base64_image_real = encode_image(real_path)
    print(f"Processing {path}")
    print(f"Description: {description}")

        
    prompt_text = f"""
You are an expert in analyzing facial images and detecting manipulations. I will provide you with two facial images — one real and one fake — along with initial detection results from the FFTG system.

The FFTG Detection result is computed as follows:
1. Computes the difference mask between real and fake images to locate manipulated regions.
2. Uses facial landmarks to analyze specific regions (eyes, nose, mouth, whole face).
3. Detects forgery artifacts in each region using:
   - **Texture abnormality**: GLCM (Gray-Level Co-occurrence Matrix) to identify unnatural texture patterns.
   - **Structural deformation**: SSIM (Structural Similarity) to detect shape distortions.
   - **Color inconsistency**: LAB color space analysis to identify unnatural color distributions.
   - **Blending artifacts**: Gradient and frequency analysis to reveal irregular region transitions.

This analysis provides a technical baseline for identifying potential manipulations.

Here are the image inputs and initial FFTG detection summary:
[Real Image]
[Fake Image]
FFTG Detection: {description}

Your task:
1. Analyze and compare the real and fake images carefully, using both FFTG results and your own expert observations.
2. For each forgery type (texture abnormality, structural deformation, color inconsistency, blending artifact), determine whether it is visible in the fake image.
3. Generate one caption for each image:
   - Captions MUST start with either **"This is a real face."** or **"This is a fake face."** (case and punctuation must match exactly).
   - Include objective, fine-grained visual analysis with supporting reasoning.
   - Do NOT copy text between captions — each must be unique in phrasing and focus.
   - Describe facial features, lighting, texture, symmetry, and any visible signs of manipulation or naturalness.
   - Include signs of uncertainty if needed (e.g., “slightly suspicious” or “mostly consistent”).

Return your response in a **JSON** format with the following fields:
- `"question"`: A natural-sounding question about the image’s authenticity. Make each question **different**. Examples include:
   - "Is this image real or fake?"
   - "Has this face been digitally altered?"
   - "Can you verify the authenticity of this facial image?"
   - "Was this face generated by AI or captured from a real person?"
   - "Is there any evidence of image tampering here?"
- "imagepath": Original Path to the image
- "answer": "real" if image is real, "fake" if it is fake
- "reason": Detailed caption and reasoning, starting with the mandatory phrase


Be sure that:
- Descriptions are not reused between the two images.
- Each description contains specific visual and analytical evidence.
- You remain objective and use signs like lighting mismatch, unnatural textures, feature asymmetries, or blending edges.
- You note uncertainty where applicable (e.g., “mostly natural, but...”).
Example Output:
[
  {{'question': 'is this face real or fake?','imagepath': '/path/real_image.png','answer': 'real','reason': 'The face is real because it has a natural appearance and shows no signs of being artificially generated or manipulated. The features are consistent with human anatomy, and there are no artifacts or distortions that would suggest it is a fake image.'}},
  {{"question":"is this face real or fake?","imagepath":"/path/fake_image.png","answer":"fake","reason":"The face is fake because it exhibits characteristics typical of computer-generated imagery, such as unnatural skin texture, unrealistic lighting, and exaggerated features. Additionally, the image may contain artifacts or inconsistencies that are not present in real human faces."}}
]

"""
    # payload = {
    #     "model": "gpt-4o",
    #     "messages": [
    #         {
    #             "role": "user",
    #             "content": [
    #                 {
    #                     "type": "text",
    #                     "text": prompt_text
    #                 },
    #                 {
    #                     "type": "image_url",
    #                     "image_url": {
    #                         "url": f"data:image/jpeg;base64,{base64_image_real}"
    #                     }
    #                 },
    #                 {
    #                     "type": "image_url",
    #                     "image_url": {
    #                         "url": f"data:image/jpeg;base64,{base64_image_fake}"
    #                     }
    #                 }
    #             ]
    #         }
    #     ],
    #     "max_tokens": 500
    # }
    client = genai.Client(api_key="AIzaSyBMHjCuXW903WWj7bdRxIikA8VmflI15Ss")

    # Upload the first image
    # image1_path = "/mnt/d/real_syntheies_datasets/rgb/FaceForensics++/manipulated_sequences/Deepfakes/c23/frames/000_003/000.png"
    # uploaded_file = client.files.upload(file=image1_path)
    image1_path = real_path
    with open(image1_path, 'rb') as f:
        img1_bytes = f.read()
    # Prepare the second image as inline data
    image2_path = path
    with open(image2_path, 'rb') as f:
        img2_bytes = f.read()
  
    max_retries = 3
    base_delay = 5  # 基础延迟时间（秒）
    for attempt in range(max_retries):
        try:
            # response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            # response.raise_for_status()  # 如果响应状态不是200，将引发HTTPError异常
            # result = response.json()["choices"][0]["message"]["content"]
              # Create the prompt with text and multiple images
            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-05-20",
                contents=[
                    prompt_text,
                    # uploaded_file,  # Use the uploaded file reference
                    types.Part.from_bytes(
                        data=img1_bytes,
                        mime_type='image/png'
                    ),
                    types.Part.from_bytes(
                        data=img2_bytes,
                        mime_type='image/png'
                    )
                ]
            )
            result=response.text
            clead_json=response.text.strip("`json\n")
            results=json.loads(clead_json)
            results[0]['imagepath']=image1_path
            results[1]['imagepath']=image2_path
            # 存储结果
            # results[path] = {
            #     "fake_path": path,
            #     "real_path": real_path,
            #     "description": description,
            #     "gpt_result": result
            # }
            
            # 立即保存结果
            save_results(results, results_file)
            
            print(f"Processed {path} finish! Caption is {result}")
            break  # 成功处理，跳出重试循环
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                delay = base_delay * (2 ** attempt)  # 指数退避
                print(f"Rate limit exceeded. Waiting for {delay} seconds before retrying.")
                time.sleep(delay)
            else:
                print(f"Error processing {path}, attempt {attempt + 1}: {str(e)}")
        
            if attempt == max_retries - 1:
                print(f"Failed to process {path} after {max_retries} attempts")
            else:
                time.sleep(5)  # 其他错误的基本等待时间


print(f"Total frames processed: {len(results)}")
print(f"Results have been saved to {results_file}")