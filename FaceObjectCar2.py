import cv2
import numpy as np
import asyncio
import websockets
import websocket
import mediapipe as mp

url = 'ws://192.168.4.1/Camera'
winName = 'ESP32 CAMERA'
cv2.namedWindow(winName, cv2.WINDOW_AUTOSIZE)

lower = np.array([25, 50, 50])
upper = np.array([30, 255, 255])

server_url = "ws://192.168.4.1/CarInput"
ws = websocket.WebSocket()

detector = mp.solutions.face_detection
cap = cv2.VideoCapture(0)

async def receive_images():
    async with websockets.connect(url) as websocket:
        while True:
            imgData = await websocket.recv()
            imgNp = np.frombuffer(imgData, dtype=np.uint8)
            img = cv2.imdecode(imgNp, -1)
            cv2.imshow(winName, img)
            if cv2.waitKey(5) & 0xFF == 27:
                break

async def detect_color_object():

    async with websockets.connect(url) as websocket:
        ws.connect(server_url)

        flag = True
        movement = "stop"

        with detector.FaceDetection(min_detection_confidence=0.75, model_selection=0) as rostros:

            while True:
                imgData = await websocket.recv()
                imgNp = np.frombuffer(imgData, dtype=np.uint8)
                video = cv2.imdecode(imgNp, -1)

                img = cv2.cvtColor(video, cv2.COLOR_BGR2HSV) 

                mask = cv2.inRange(img, lower, upper)  

                mask_contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  

                if len(mask_contours) != 0:
                    for mask_contour in mask_contours:
                        if cv2.contourArea(mask_contour) > 6000:
                            x, y, w, h = cv2.boundingRect(mask_contour)
                            cv2.rectangle(video, (x, y), (x + w, y + h), (50, 255, 50), 3)  
                            flag = False
                            movement = "left"
                            print("Objeto detectado")
                            mensaje = "MoveCar, 3"  
                            ws.send(mensaje)

                if flag:

                    ret, frame = cap.read()
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    resultado = rostros.process(rgb)
                    listarostro = []

                    if resultado.detections is not None:
                        for rostro in resultado.detections:
                            for id, puntos in enumerate(resultado.detections):
                                al, an, _ = frame.shape
                                # print("Alto: ", al, "Ancho: ", an)
                                centro = int(an / 2)
                                centroy = int(al / 2)
                                x = puntos.location_data.relative_bounding_box.xmin
                                y = puntos.location_data.relative_bounding_box.ymin
                                ancho = puntos.location_data.relative_bounding_box.width
                                alto = puntos.location_data.relative_bounding_box.height
                                x, y = int(x * an), int(y * al)
                                # print("X, Y: ", x, y)
                                x1, y1 = int(ancho * an), int(alto * al)
                                xf, yf = x + x1, y + y1
                                cx = (x + (x + x1)) // 2
                                cy = (y + (y + y1)) // 2
                                listarostro.append([x, y, x1, y1])
                                cv2.line(frame, (cx, 0), (cx, 720), (0, 255, 0), 2)
                                cv2.line(frame, (0, cy), (1280, cy), (0, 255, 0), 2)
                                cv2.rectangle(frame, (x, y), (xf, yf), (255, 255, 0), 3)
                                cv2.namedWindow('Camara')
                                if cx < centro - 50 and movement != "left":
                                    movement = "left"
                                    mensaje = "MoveCar, 3"  
                                    print(mensaje, "Izquierda")
                                    ws.send(mensaje)
                                elif cx > centro + 50 and movement != "right":
                                    movement = "right"
                                    mensaje = "MoveCar, 4" 
                                    print(mensaje, "Derecha") 
                                    ws.send(mensaje)
                                if cy < centroy - 50 and centro - 50 < cx < centro + 50 and movement != "forward":
                                    movement = "forward"
                                    mensaje = "MoveCar, 1"  
                                    print(mensaje, "Adelante")
                                    ws.send(mensaje)
                                elif cy > centroy + 50 and centro - 50 < cx < centro + 50 and movement != "backward":
                                    movement = "backward"
                                    mensaje = "MoveCar, 2" 
                                    print(mensaje, "Atras") 
                                    ws.send(mensaje)
                                elif cx > centro - 50 and cx < centro + 50 and cy > centroy - 50 and cy < centroy + 50 and movement != "stop":
                                    movement = "stop"
                                    mensaje = "MoveCar, 0" 
                                    print(mensaje, "Detenido") 
                                    ws.send(mensaje)
                    cv2.imshow('Camara', frame)
                flag = True

                cv2.imshow("window", video) 
                cv2.waitKey(50)

asyncio.get_event_loop().run_until_complete(asyncio.gather(receive_images(), detect_color_object()))
cv2.destroyAllWindows()
