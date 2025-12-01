import os
import sys
import re
import base64
import io
import zipfile
import shutil
import uuid
from datetime import datetime

import dash
from dash import dcc, html, Input, Output, State, register_page

register_page(__name__)

# --- 기존 로직 (main.py에서 가져옴) ---
def process_file_recursively(filepath, output_file, processed_files, base_dir):
    """
    파일을 재귀적으로 읽고 'INSERT' 지시어를 처리하여 output_file에 씁니다.
    웹 앱 환경에 맞게 base_dir 인자를 추가했습니다.
    """
    abs_filepath = os.path.abspath(filepath)

    if abs_filepath in processed_files:
        print(f"경고: 순환 참조가 감지되어 '{filepath}' 파일을 다시 삽입하지 않습니다.", file=sys.stderr)
        return

    if not os.path.exists(abs_filepath):
        print(f"오류: '{filepath}' 파일을 찾을 수 없습니다.", file=sys.stderr)
        return

    print(f"처리 중: '{filepath}'")
    processed_files.add(abs_filepath)

    current_dir = os.path.dirname(abs_filepath)

    try:
        with open(abs_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                stripped_line = line.strip()
                comment_pos = stripped_line.find(';')
                effective_line = stripped_line.split(';', 1)[0].strip() if comment_pos != -1 else stripped_line

                match = re.match(r'^INSERT\s+([^\s]+)', effective_line)
                if match:
                    filename_to_insert = match.group(1).strip()
                    if '.' not in filename_to_insert:
                        filename_to_insert += '.asc'
                    
                    path_to_insert = os.path.join(base_dir, filename_to_insert)
                    
                    process_file_recursively(path_to_insert, output_file, processed_files, base_dir)
                    output_file.write('\n')
                else:
                    output_file.write(line)
    except Exception as e:
        print(f"오류: '{filepath}' 파일을 읽는 중 예외가 발생했습니다: {e}", file=sys.stderr)

    processed_files.remove(abs_filepath)

def merge_files(main_input_file, output_filename, base_dir):
    """
    메인 함수: 파일 병합 프로세스를 시작합니다.
    """
    processed_files = set()
    try:
        with open(output_filename, 'w', encoding='utf-8') as output_f:
            print(f"병합 시작: '{main_input_file}' -> '{output_filename}'")
            process_file_recursively(main_input_file, output_f, processed_files, base_dir)
        print("병합 완료!")
        return True
    except Exception as e:
        print(f"오류: 최종 파일을 쓰는 중 예외가 발생했습니다: {e}", file=sys.stderr)
        return False

# --- Dash 애플리케이션 설정 ---

#app = dash.Dash(__name__)
#server = app.server

layout = html.Div(
    style={'fontFamily': 'Arial, sans-serif', 'maxWidth': '1000px', 'margin': 'auto', 'padding': '20px'},
    id='app-container', children=[
    # 이 페이지에만 특정 CSS 파일을 적용하기 위해 html.Link를 추가합니다.
    # CSS 파일은 'assets' 폴더 내에 있어야 합니다.
    html.Link(
        rel='stylesheet',
        href='/assets/style.css'
    ),

    html.Div(className='header', children=[
        html.H1("ATL 패턴 추출기"),
        html.P("파일 병합을 위해 ZIP 파일을 업로드하고 Pattern 이름(ex: pattern.asc)을 입력하세요."),
    ]),

    html.Div(className='controls-container', children=[
        dcc.Upload(
            id='upload-zip',
            children=html.Div([
                'ZIP 파일을 드래그 앤 드롭하거나 ',
                html.A('파일을 선택하세요')
            ]),
            multiple=False
        ),
        html.Div(id='upload-status'),
        
        dcc.Input(
            id='main-file-name',
            type='text',
            placeholder='변경할 Pattern 파일 이름 (예: pattern.asc)',
        ),
        
        html.Button('병합 실행', id='submit-button', n_clicks=0),
    ]),
    html.Div(id='output-status', className='status-container'),
    dcc.Download(id="download-result"),
])

@dash.callback(
    Output('upload-status', 'children'),
    Input('upload-zip', 'filename')
)
def update_upload_status(filename):
    if filename:
        return html.Div(f"'{filename}' 파일이 로드되었습니다.", className='upload-success')
    return ""

@dash.callback(
    [Output('output-status', 'children'),
     Output('download-result', 'data')],
    [Input('submit-button', 'n_clicks')],
    [State('upload-zip', 'contents'),
     State('upload-zip', 'filename'),
     State('main-file-name', 'value')]
)
def update_output(n_clicks, content, zip_filename, main_filename):
    if n_clicks == 0 or not content:
        return "", None

    if not main_filename:
        return html.Div("Pattern 파일 이름을 입력하세요.", className='status-error'), None

    # 임시 작업 디렉토리 생성
    session_id = str(uuid.uuid4())
    work_dir = os.path.join(os.getcwd(), "temp", session_id)
    os.makedirs(work_dir, exist_ok=True)

    try:
        # ZIP 파일 처리
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        
        with zipfile.ZipFile(io.BytesIO(decoded), 'r') as zip_ref:
            zip_ref.extractall(work_dir)

        # ZIP 파일 내에 최상위 폴더가 있는 경우 처리
        extracted_items = os.listdir(work_dir)
        if len(extracted_items) == 1 and os.path.isdir(os.path.join(work_dir, extracted_items[0])):
            # 압축 해제된 내용이 단일 폴더 안에 있을 경우, 내용물을 work_dir의 루트로 이동
            nested_dir = os.path.join(work_dir, extracted_items[0])
            for item in os.listdir(nested_dir):
                shutil.move(os.path.join(nested_dir, item), work_dir)
            os.rmdir(nested_dir)

        # 메인 파일이 ZIP 파일의 루트에 있는지 대소문자를 구분하여 확인
        file_list_in_zip = os.listdir(work_dir)
        if main_filename not in file_list_in_zip:
            # ZIP 파일 내의 파일 목록을 보여주어 디버깅에 도움을 줌
            file_list_for_error = [f for f in file_list_in_zip if os.path.isfile(os.path.join(work_dir, f))]
            return html.Div(f"오류: ZIP 파일 내에서 '{main_filename}'을(를) 찾을 수 없습니다. (대소문자 구분) 사용 가능한 파일: {', '.join(file_list_for_error)}", className='status-error'), None

        # 병합할 파일 경로 설정
        main_file_path = os.path.join(work_dir, main_filename)

        # 출력 파일 경로 설정
        # rsplit을 사용하여 오른쪽부터 '.'을 기준으로 1번만 분리합니다.
        parts = main_filename.rsplit('.', 1)
        output_filename = parts[0] + '_new' + '.' + parts[1]
        output_path = os.path.join(work_dir, output_filename)

        # 파일 병합 실행
        success = merge_files(main_file_path, output_path, work_dir)

        if success and os.path.exists(output_path):
            # 성공 시 다운로드 준비
            status_message = html.Div([
                html.P(f"병합 성공! '{output_filename}' 파일을 다운로드합니다."),
                html.P(f"(업로드된 파일: {zip_filename}, 메인 파일: {main_filename})")
            ], className='status-success')
            
            return status_message, dcc.send_file(output_path, filename=output_filename)
        else:
            # 실패 시 오류 메시지
            return html.Div("파일 병합 중 오류가 발생했습니다. 서버 로그를 확인하세요.", className='status-error'), None

    except Exception as e:
        return html.Div(f"처리 중 예외가 발생했습니다: {e}", className='status-error'), None
    finally:
        # 임시 디렉토리 정리 (선택적: 디버깅을 위해 남겨둘 수 있음)
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)