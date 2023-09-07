import cv2
import numpy as np
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import base64

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("RGB Feature Display"),
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drop or   ',
            html.A('Select Image')
        ]),
        style={
            'width': '50%',
            'height': '100px',
            'lineHeight': '100px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    html.Div(id='output-image-upload')
])

def analyze_image(contents):
    # 将上传的内容解码为图像格式
    encoded_image = contents.split(',')[1]
    decoded_image = cv2.imdecode(np.frombuffer(
        bytes(base64.b64decode(encoded_image)), np.uint8), -1)

    # 分别获取三个通道的ndarray数据
    img_b = decoded_image[:,:,0]
    img_g = decoded_image[:,:,1]
    img_r = decoded_image[:,:,2]

    # 按R、G、B三个通道分别计算颜色直方图
    b_hist = cv2.calcHist([decoded_image],[0],None,[256],[0,255])
    g_hist = cv2.calcHist([decoded_image],[1],None,[256],[0,255])
    r_hist = cv2.calcHist([decoded_image],[2],None,[256],[0,255])
    m, dev = cv2.meanStdDev(decoded_image)  # 计算G、B、R三通道的均值和方差

    # 计算三个通道的均值和标准差
    r_mean, r_std = np.mean(r_hist), np.std(r_hist)
    g_mean, g_std = np.mean(g_hist), np.std(g_hist)
    b_mean, b_std = np.mean(b_hist), np.std(b_hist)
    m = m.ravel()
    dev = dev.ravel()
    return r_mean, r_std, g_mean, g_std, b_mean, b_std,m,dev

@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'))
def update_output_image_upload(contents):
    if contents is not None:
        r_mean, r_std, g_mean, g_std, b_mean, b_std,m,dev = analyze_image(contents)

        # 将图像显示在网页上
        return html.Div([
            html.H3('Uploaded image:'),
            html.Img(src=contents, style={'width': '400px'}),
            #html.P(f'R通道均值：{r_mean:.2f}，标准差：{r_std:.2f}'),
            #html.P(f'G通道均值：{g_mean:.2f}，标准差：{g_std:.2f}'),
            #html.P(f'B通道均值：{b_mean:.2f}，标准差：{b_std:.2f}'),
            html.P(f'Rmean：{m.ravel().tolist()[2]:.2f}，R standard deviation：{dev.ravel().tolist()[2]:.2f}'),
            html.P(f'Gmean：{m.ravel().tolist()[1]:.2f}，G standard deviation：{dev.ravel().tolist()[1]:.2f}'),
            html.P(f'Bmean：{m.ravel().tolist()[0]:.2f}，B standard deviation：{dev.ravel().tolist()[0]:.2f}'),
        ])

if __name__ == '__main__':
    app.run_server(debug=False)