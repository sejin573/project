pip install openai==0.28
import streamlit as st
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input as inception_preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_preprocess_input
from tensorflow.keras.applications.efficientnet import EfficientNetB2, preprocess_input as efficientnet_preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import openai
import base64
import os

# ChatGPT API 인증 설정
openai.api_key = st.secrets["api_key"]
# 페이지 너비 조정
st.set_page_config(layout="wide")

# 모델 선택
def select_model():
    models = {
        'ResNet50': (ResNet50, (224, 224)),
        'InceptionV3': (InceptionV3, (299, 299)),
        'MobileNetV2': (MobileNetV2, (224, 224)),
        'EfficientNetV2S': (EfficientNetB2, (260, 260))
    }
    model_name = st.sidebar.selectbox('모델 선택', list(models.keys()))
    model_func, input_shape = models[model_name]
    model = model_func(weights='imagenet')
    return model, input_shape, model_name

# 이미지 분류
def classify_image(model, input_shape, model_name, file_path):
    img = image.load_img(file_path, target_size=input_shape)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    if model_name == 'ResNet50':
        x = preprocess_input(x)
    elif model_name == 'InceptionV3':
        x = inception_preprocess_input(x)
    elif model_name == 'MobileNetV2':
        x = mobilenet_preprocess_input(x)
    elif model_name == 'EfficientNetV2S':
        x = efficientnet_preprocess_input(x)
    else:
        raise ValueError(f"지원되지 않는 모델입니다: {model_name}")
    
    preds = model.predict(x)
    
    if model_name == 'ResNet50':
        results = decode_predictions(preds, top=3)[0]
    elif model_name == 'InceptionV3':
        results = decode_predictions(preds, top=3)[0]
    elif model_name == 'MobileNetV2':
        results = decode_predictions(preds, top=3)[0]
    elif model_name == 'EfficientNetV2S':
        results = decode_predictions(preds, top=3)[0]
    else:
        raise ValueError(f"지원되지 않는 모델입니다: {model_name}")
    
    return results

# ChatGPT 대화
def chat_with_gpt(message):
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=message,
        max_tokens=700,
        temperature=0.7,
        n=1,
        stop=None
    )
    reply = response.choices[0].text.strip()
    return reply

# 분류 결과값 설명 출력
def display_class_definition(class_label):
    gpt_message = f"이미지 클래스 {class_label}에 대한 정의는 무엇인가요?"
    class_definition = chat_with_gpt(gpt_message)

    st.sidebar.markdown(f"**{class_label}**")
    st.sidebar.warning(f"설명: {class_definition}")

# 이미지 갤러리 표시
def display_image_gallery(images, model, input_shape, model_name):
    num_images = len(images)
    cols = st.columns(3)
    index = 0
    for i in range((num_images + 2) // 3):
        for j in range(3):
            if index < num_images:
                cols[j].subheader(f"이미지 {index+1}")
                cols[j].image(images[index], use_column_width=True)
                if cols[j].button("분류하기", key=f"classify_button_{index}"):
                    # 이미지 분류 수행
                    results = classify_image(model, input_shape, model_name, images[index])
                    cols[j].subheader(f"이미지 {index+1} - 분류 결과")
                    for result in results:
                        class_label = result[1]
                        confidence = result[2] * 100
                        cols[j].write(f"{class_label}: {confidence:.2f}%")
                        display_class_definition(class_label)
                    cols[j].write("---")
            index += 1

# 웹페이지 구현
def main():
    st.title('졸업 프로젝트 인공지능 웹페이지')

    # 배경색 설정
    st.markdown(
        """
        <style>
        .reportview-container {
            background: linear-gradient(#E4E5E6, #E4E5E6);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # 왼쪽 컬럼
    st.sidebar.subheader("기능")
    function = st.sidebar.selectbox("", ["이미지 분류", "챗GPT"])

    # 이미지 분류 기능
    if function == "이미지 분류":
        model, input_shape, model_name = select_model()

        # 이미지 업로드
        st.subheader("이미지 업로드")
        uploaded_files = st.file_uploader("여러 이미지 업로드", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        if uploaded_files is not None:
            # 분류 작업 수행
            images = []
            for uploaded_file in uploaded_files:
                images.append(uploaded_file)

            # 이미지 갤러리 표시
            display_image_gallery(images, model, input_shape, model_name)

    # 챗GPT와 대화하기
    if function == "챗GPT":
        st.subheader("챗GPT와 대화하기")
        question_input = st.text_input("질문을 입력하세요.")
        if question_input:
            gpt_reply = chat_with_gpt(question_input)
            st.write(f"질문: {question_input}")
            st.write(f"답변: {gpt_reply}")


if __name__ == '__main__':
    main()
