import streamlit as st
import torch
import os
import json
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

# === OpenAI-related library (import according to your environment) ===
from openai import OpenAI

# === CreatiLayout utility/model loading ===
import sys
sys.path.append("./src/CreatiLayout")
API_KEY = os.getenv("API_KEY")

from src.CreatiLayout.utils.bbox_visualization import bbox_visualization, scale_boxes
from src.CreatiLayout.src.models.transformer_sd3_SiamLayout import SiamLayoutSD3Transformer2DModel
from src.CreatiLayout.src.pipeline.pipeline_CreatiLayout import CreatiLayoutSD3Pipeline

# --------------------------------------------------------------------------------
# Default GPT "system" instruction (can be edited via the expander in the UI)
# --------------------------------------------------------------------------------
default_system_instruction = (
    "You are an assistant that designs layouts for movie thumbnails. "
    "The layout should have between three and five objects. Each object should be described "
    "with (object name, a short feature, [x1, y1, x2, y2]). "
    "Additionally, include one line that describes the overall mood/background "
    "of the entire image and also provide a 'title' for the movie thumbnail. "
    "The coordinates x, y, w, h should be between 0 and 1. Output in valid JSON format with the schema:\n"
    "{\n"
    '  "title": "...",\n'
    '  "layout": {\n'
    '    "objects": [\n'
    '      {\n'
    '        "object_name": "...",\n'
    '        "feature": "...",\n'
    '        "bbox": [x1, y1, x2, y2]\n'
    "      }, ...\n"
    "    ],\n"
    '    "overall_mood": "..." \n'
    "  }\n"
    "}\n"
)

# --------------------------------------------------------------------------------
# 1) Function to request "movie thumbnail layout" from GPT
# --------------------------------------------------------------------------------
def get_layout_from_gpt(api_key: str, scenario: str, system_instruction: str) -> dict:
    """
    Sends the scenario to OpenAI GPT and receives the thumbnail layout in JSON format.
    """
    # Create a ChatGPT-like client
    client = OpenAI(api_key=api_key)

    # Compose system and user messages
    messages = [
        {
            "role": "system",
            "content": system_instruction
        },
        {
            "role": "user",
            "content": (
                "[Instruction] Please create a layout for a movie thumbnail.\n\n"
                "The product of the bounding box's width and height must be greater than 0.16.\n"
                f"Here is the scenario:\n{scenario}"
            )
        },
    ]

    # ChatCompletion request
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Modify to a real model that you can use
        messages=messages
    )

    layout_response = response.choices[0].message.content

    # Attempt to extract JSON from the response
    start_index = layout_response.find('{')
    end_index = layout_response.rfind('}')
    if start_index != -1 and end_index != -1 and start_index < end_index:
        layout_response = layout_response[start_index:end_index + 1]

    # Parse JSON
    try:
        layout_data = json.loads(layout_response)
    except json.JSONDecodeError:
        raise ValueError("The ChatGPT response is not in valid JSON format:\n" + layout_response)

    return layout_data


# --------------------------------------------------------------------------------
# 2) Load models (stable paths, no user input for model/ckpt)
# --------------------------------------------------------------------------------
@st.cache_resource
def load_models(device):
    """
    Load Stable Diffusion + Layout Transformer models.
    """
    # Load the Transformer
    transformer_additional_kwargs = dict(attention_type="layout", strict=True)
    transformer = SiamLayoutSD3Transformer2DModel.from_pretrained(
        "HuiZhang0812/CreatiLayout",  # fixed path
        subfolder="transformer",
        torch_dtype=torch.float16,
        **transformer_additional_kwargs
    )

    # Load the Pipeline
    pipe = CreatiLayoutSD3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",  # fixed path
        transformer=transformer,
        torch_dtype=torch.float16
    )
    pipe = pipe.to(device)
    return pipe


# --------------------------------------------------------------------------------
# 3) Generate image
# --------------------------------------------------------------------------------
def generate_image(pipe, params, image_placeholder):
    """
    수정된 generate_image 함수:
    - callback_on_step_end를 설정하여 5 스텝마다 중간 결과를 디코딩하고,
      단일 Streamlit 이미지 placeholder를 업데이트.
    - 최종 이미지는 기존처럼 반환.
    """
    
    import torch
    from PIL import Image
    import streamlit as st

    # (1) callback 함수 정의
    def on_step_end(pipeline, step, t, callback_kwargs):
        """
        매 스텝이 끝날 때마다 자동으로 불리는 함수.
        step(현재 몇 번째 스텝인지), t(timestep 값), callback_kwargs에 latents 등 포함.
        """
        latents = callback_kwargs["latents"]  # pipeline 내부의 latents
        # 5 스텝마다만 중간 이미지를 디코딩 & 표시
        if step % 5 == 0 and step > 0:
            # VAE로 디코딩
            latents_for_decode = (latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
            image_torch = pipeline.vae.decode(latents_for_decode, return_dict=False)[0]
            image_pil = pipeline.image_processor.postprocess(image_torch, output_type="pil")
            
            # Streamlit의 단일 이미지 placeholder 업데이트
            image_placeholder.image(image_pil, caption=f"Step {step} / {params['num_inference_steps']}", use_column_width=True)
        
        # 콜백은 dict를 리턴할 수 있으며, 여기서는 latents를 수정하지 않음
        return {"latents": latents}

    # (2) pipe 호출 시 callback 전달
    with torch.no_grad():
        output = pipe(
            prompt=params['prompt'],
            num_inference_steps=params['num_inference_steps'],
            guidance_scale=params['guidance_scale'],
            bbox_phrases=params['region_caption_list'],
            bbox_raw=params['region_bboxes_list'],
            height=params['height'],
            width=params['width'],

            # 콜백 관련 인자
            callback_on_step_end=on_step_end,
            callback_on_step_end_tensor_inputs=["latents"]
        )

    # output.images (기본 1장이라고 가정)
    final_image = output.images[0]
    image_placeholder.image(final_image, caption=f"Final Generation", use_column_width=True)
    
    return final_image


# --------------------------------------------------------------------------------
# 4) Save and visualize the generated image with bounding boxes
# --------------------------------------------------------------------------------
def save_and_visualize_image(image, params):
    """
    Save the generated image and visualize bounding boxes on it,
    then display them on Streamlit.
    """
    # Prepare output folders
    os.makedirs(params['img_save_root'], exist_ok=True)
    os.makedirs(params['img_with_layout_save_root'], exist_ok=True)

    # Save the original image
    img_path = os.path.join(params['img_save_root'], f"{params['filename']}.png")
    image.save(img_path)

    # Prepare bounding box visualization
    numeric_labels = [f"({i+1})" for i in range(len(params['region_caption_list']))]
    show_input = {
        "boxes": scale_boxes(params['region_bboxes_list'], params['width'], params['height']),
        "labels": numeric_labels
    }

    # Create a white background image of the same size
    white_image = Image.new('RGB', (params['width'], params['height']), color='rgb(255,255,255)')
    bbox_visualization_img = bbox_visualization(white_image, show_input)
    image_with_bbox = bbox_visualization(image, show_input)

    # Concatenate the two images horizontally
    total_width = params['width'] * 2
    total_height = params['height']
    new_image = Image.new('RGB', (total_width, total_height))
    new_image.paste(bbox_visualization_img, (0, 0))
    new_image.paste(image_with_bbox, (params['width'], 0))

    # Save the concatenated image
    img_with_layout_save_name = os.path.join(
        params['img_with_layout_save_root'], f"{params['filename']}_layout.png"
    )
    new_image.save(img_with_layout_save_name)

    # Display in Streamlit (original vs. with layout)
    st.image(
        [Image.open(img_with_layout_save_name)], 
        caption=["With Layout"], 
        use_column_width=True
    )

    # Display object legend
    st.markdown("### Object Legend")
    for idx, label in enumerate(numeric_labels):
        st.markdown(f"""
            <span style="font-size: 16px; font-weight: bold; color: #4CAF50;">{label}</span> → 
            <span style="font-size: 16px; font-weight: bold;">{params['region_caption_list'][idx]}</span>
        """, unsafe_allow_html=True)   
        #st.write(f"{label} → {params['region_caption_list'][idx]}")


# --------------------------------------------------------------------------------
# (Additional) Preview layout on a white background
# --------------------------------------------------------------------------------
def visualize_layout_on_white(layout_data, preview_width=512, preview_height=512):
    """
    Takes layout_data from GPT (or user-modified) and visualizes it on a white background.
    Bbox labels are in the form (1), (2), (3)... for simplicity.
    """
    white_image = Image.new('RGB', (preview_width, preview_height), color='white')
    objects = layout_data["layout"]["objects"]

    # Scale bboxes from [0,1] range to actual pixel size
    scaled_bboxes = scale_boxes(
        [obj["bbox"] for obj in objects], 
        preview_width, 
        preview_height
    )

    numeric_labels = [f"({i+1})" for i in range(len(objects))]

    show_input = {
        "boxes": scaled_bboxes,
        "labels": numeric_labels
    }
    vis_img = bbox_visualization(white_image, show_input)
    return vis_img, numeric_labels


# --------------------------------------------------------------------------------
# 5) Streamlit main app
# --------------------------------------------------------------------------------
def main():
    st.title("🚀 Movie Thumbnail Layout + Image Generator 🌟")
    st.write("1) Input your scenario → GPT generates a layout → 2) Modify/Preview layout → 3) Generate the final image")

    # === (A) GPT System Instruction (editable via expander) ===
    with st.expander("Edit GPT System Instruction (click to expand)", expanded=False):
        system_instruction_input = st.text_area(
            "GPT System Instruction", 
            default_system_instruction, 
            height=600
        )

    # === (B) OpenAI API Key input ===
    api_key = st.text_input("Please enter your OpenAI API Key here:", 
                            value =API_KEY,
                            type="password")

    # === (C) Scenario input ===
    scenario = st.text_area(
        "Enter your movie scenario or summary here:", 
        "Princess Leia is captured and held hostage... restore peace and justice in the Empire."
    )

    # Prepare a place to store the layout data in session
    if 'layout_data' not in st.session_state:
        st.session_state['layout_data'] = None

    # -------------------------------------------------------------------------
    # (1) Request layout from GPT
    # -------------------------------------------------------------------------
    if st.button("1) Generate Layout"):
        if not api_key:
            st.warning("Please provide your OpenAI API Key first.")
        else:
            with st.spinner("Requesting layout from GPT..."):
                layout_data = get_layout_from_gpt(api_key, scenario, system_instruction_input)
                st.session_state['layout_data'] = layout_data
            st.success("Layout creation complete!")
            with st.expander("View Layout JSON (click to expand)", expanded=False):
                st.json(st.session_state['layout_data'])

    # -------------------------------------------------------------------------
    # (1.5) Layout preview and modification
    # -------------------------------------------------------------------------
    if st.session_state['layout_data']:

        st.subheader("Layout Preview and Modification")

        # Default preview size
        preview_width = 512
        preview_height = 512

        # Visualize current layout
        layout_preview, numeric_labels = visualize_layout_on_white(
            st.session_state['layout_data'], 
            preview_width=preview_width, 
            preview_height=preview_height
        )
        st.image(layout_preview, caption="Current Layout Preview (on white)", use_column_width=True)

        # Display object legend
        st.markdown("**[Object Legend (Current Layout)]**")
        for i, obj in enumerate(st.session_state['layout_data']["layout"]["objects"]):
            st.markdown(f"""
                <span style="font-size: 16px; font-weight: bold; color: #4CAF50;">{numeric_labels[i]}</span> → 
                <span style="font-size: 16px; font-weight: bold;">{obj['object_name']}</span> / 
                <span style="font-size: 14px; color: #757575;">{obj['feature']}</span>
            """, unsafe_allow_html=True)

        # -- Modify layout info --
        layout_data = st.session_state['layout_data']
        title = st.text_input("Thumbnail Title", layout_data["title"])
        overall_mood = st.text_input("Overall Mood", layout_data["layout"]["overall_mood"])

        updated_objects = []
        for idx, obj in enumerate(layout_data["layout"]["objects"]):
            st.markdown(f"**[Object {idx+1}]**  {numeric_labels[idx]}")
            col_left, col_right = st.columns([2, 2])
            with col_left:
                object_name = st.text_input(
                    f"Object Name #{idx+1}",
                    obj["object_name"], 
                    key=f"object_name_{idx}"
                )
                feature = st.text_input(
                    f"Object Feature #{idx+1}", 
                    obj["feature"], 
                    key=f"feature_{idx}"
                )
            with col_right:
                x1 = st.number_input(
                    f"x1 (#{idx+1})", 
                    0.0, 1.0, 
                    value=float(obj["bbox"][0]), 
                    step=0.01, 
                    key=f"x1_{idx}"
                )
                y1 = st.number_input(
                    f"y1 (#{idx+1})", 
                    0.0, 1.0, 
                    value=float(obj["bbox"][1]), 
                    step=0.01, 
                    key=f"y1_{idx}"
                )
                x2 = st.number_input(
                    f"x2 (#{idx+1})", 
                    0.0, 1.0, 
                    value=float(obj["bbox"][2]), 
                    step=0.01, 
                    key=f"x2_{idx}"
                )
                y2 = st.number_input(
                    f"y2 (#{idx+1})", 
                    0.0, 1.0, 
                    value=float(obj["bbox"][3]), 
                    step=0.01, 
                    key=f"y2_{idx}"
                )

            updated_objects.append({
                "object_name": object_name,
                "feature": feature,
                "bbox": [x1, y1, x2, y2]
            })

        # -- Update Layout button --
        if st.button("Update Layout"):
            st.session_state['layout_data']["title"] = title
            st.session_state['layout_data']["layout"]["overall_mood"] = overall_mood
            st.session_state['layout_data']["layout"]["objects"] = updated_objects

            st.success("Layout updated.")

            # Preview the updated layout
            layout_preview, new_numeric_labels = visualize_layout_on_white(
                st.session_state['layout_data'], 
                preview_width=preview_width, 
                preview_height=preview_height
            )
            st.image(layout_preview, caption="Preview of the Updated Layout", use_column_width=True)

            st.markdown("**[Object Legend (Updated)]**")
            for i, obj in enumerate(st.session_state['layout_data']["layout"]["objects"]):
                st.markdown(f"""
                    <span style="font-size: 16px; font-weight: bold; color: #4CAF50;">{numeric_labels[i]}</span> → 
                    <span style="font-size: 16px; font-weight: bold;">{obj['object_name']}</span> / 
                    <span style="font-size: 14px; color: #757575;">{obj['feature']}</span>
                """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # (2) Model loading & image generation
    # -------------------------------------------------------------------------
    st.write("---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with st.spinner("Loading the model..."):
        pipe = load_models(device)
    st.success("Model loaded successfully!")

    st.subheader("Image Generation Parameters")

    num_inference_steps = st.number_input("Number of Inference Steps", min_value=1, max_value=500, value=50)
    guidance_scale = st.slider("Guidance Scale", min_value=0.0, max_value=20.0, value=7.5)
    height = st.number_input("Image Height (px)", min_value=64, max_value=2048, value=512)
    width = st.number_input("Image Width (px)", min_value=64, max_value=2048, value=512)
    filename = st.text_input("Filename for Saving the Generated Image", "movie_thumbnail")

    img_save_root = os.path.join("output", "images")
    img_with_layout_save_root = os.path.join("output", "images_with_layout")

    # === Generate image button ===
    if st.button("2) Generate Image"):
        if not st.session_state['layout_data']:
            st.warning("Please generate or update the layout first.")
        else:
            layout_data = st.session_state['layout_data']

            # Create a global prompt (e.g., combining Title + Overall Mood)
            global_prompt = f"{layout_data['title']} / {layout_data['layout']['overall_mood']}"

            # Object-level captions and bounding boxes
            region_caption_list = []
            region_bboxes_list = []
            for obj in layout_data['layout']['objects']:
                region_caption_list.append(f"{obj['object_name']} {obj['feature']}")
                region_bboxes_list.append(obj['bbox'])

            # Prepare parameters for the pipeline call
            params = {
                'prompt': global_prompt,
                'num_inference_steps': int(num_inference_steps),
                'guidance_scale': float(guidance_scale),
                'region_caption_list': region_caption_list,
                'region_bboxes_list': region_bboxes_list,
                'height': int(height),
                'width': int(width),
                'img_save_root': img_save_root,
                'img_with_layout_save_root': img_with_layout_save_root,
                'filename': filename,
            }

            # === 이미지 업데이트를 위한 placeholder 생성 ===
            image_placeholder = st.empty()
            with st.spinner("Generating the image..."):
                # 이미지 생성
                final_image = generate_image(pipe, params, image_placeholder)

                # 최종 이미지 저장 및 시각화
                save_and_visualize_image(final_image, params)
            st.success("Image generation and visualization complete!")

    st.write("---")
    st.subheader("Output Path Information")
    st.write(f"- Original images: `{img_save_root}`")
    st.write(f"- Images with layout visualization: `{img_with_layout_save_root}`")
    
            # st.write("**Image Diffusion Process Static Visualize**")
            # for step_idx, im in intermediate_images:
            #     st.write(f"Step {step_idx}")
            #     st.image(im, width=256)
    # # === Generate image button ===
    # if st.button("2) Generate Image"):
    #     if not st.session_state['layout_data']:
    #         st.warning("Please generate or update the layout first.")
    #     else:
    #         layout_data = st.session_state['layout_data']

    #         # Create a global prompt (e.g., combining Title + Overall Mood)
    #         global_prompt = f"{layout_data['title']} / {layout_data['layout']['overall_mood']}"

    #         # Object-level captions and bounding boxes
    #         region_caption_list = []
    #         region_bboxes_list = []
    #         for obj in layout_data['layout']['objects']:
    #             region_caption_list.append(f"{obj['object_name']} {obj['feature']}")
    #             region_bboxes_list.append(obj['bbox'])

    #         # Prepare parameters for the pipeline call
    #         params = {
    #             'prompt': global_prompt,
    #             'num_inference_steps': int(num_inference_steps),
    #             'guidance_scale': float(guidance_scale),
    #             'region_caption_list': region_caption_list,
    #             'region_bboxes_list': region_bboxes_list,
    #             'height': int(height),
    #             'width': int(width),
    #             'img_save_root': img_save_root,
    #             'img_with_layout_save_root': img_with_layout_save_root,
    #             'filename': filename,
    #         }

    #         with st.spinner("Generating the image..."):
    #             result_image = generate_image(pipe, params)
    #             save_and_visualize_image(result_image, params)
    #         st.success("Image generation and visualization complete!")

if __name__ == "__main__":
    main()
