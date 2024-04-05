import cv2
import argparse
from collections import defaultdict
from tkinter import Tk,  Label, Button
from ultralytics import YOLO
import supervision as sv
import numpy as np
import webbrowser
import tkinter
import time

def activador():
    ZONE_POLYGON = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
    ])


    def parse_arguments() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="YOLOv8 live")
        parser.add_argument(
            "--webcam-resolution",
            default=[1280, 720], 
            nargs=2, 
            type=int
        )
        args = parser.parse_args()
        return args


    def main():
        args = parse_arguments()
        frame_width, frame_height = args.webcam_resolution

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

        model = YOLO("dato.pt")

        box_annotator = sv.BoxAnnotator(
            thickness=2,
            text_thickness=2,
            text_scale=1
        )

        zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
        zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=tuple(args.webcam_resolution))
        zone_annotator = sv.PolygonZoneAnnotator(
            zone=zone, 
            color=sv.Color.red(),
            thickness=2,
            text_thickness=4,
            text_scale=2
        )

        class_counts = defaultdict(int)
        last_detection_time = time.time()
        detection_interval = 0.5
        while True:
         ret, frame = cap.read()
         if time.time() - last_detection_time >= detection_interval:
            result = model(frame, agnostic_nms=True)[0]
            detections = sv.Detections.from_yolov8(result)
            labels = [
                f"{model.model.names[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, _
                in detections
            ]
            frame = box_annotator.annotate(
                scene=frame, 
                detections=detections, 
                labels=labels
            )

            zone.trigger(detections=detections)
            frame = zone_annotator.annotate(scene=frame)

            for _, _, class_id, _ in detections:
                class_counts[class_id] += 1

            for class_id, count in class_counts.items():
                cv2.putText(
                    frame, 
                    f"{model.model.names[class_id]}: {count}", 
                    (10, 30 + class_id * 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 0, 255), 
                    2
                )
            last_detection_time = time.time()
            cv2.imshow("(Detector de Valvulas)", frame)

            if (cv2.waitKey(30) == 27):
                break


    if __name__ == "__main__":
        main()

def Contacto():
    ventana = Tk ()
    ventana.geometry("700x280")
    ventana.title("Calculador")

    lbl = Label(ventana, text=" Numero de contacto ")
    lbl.pack()
    
    lbl = Label(ventana, text=" Correo electronico ")
    lbl.pack()
    
    lbl = Label(ventana, text=" Brochure ")
    lbl.pack()
    
    btn = Button(ventana, text="Impresion de Info", command=Contacto)
    btn.pack()

def Facebook():
    url = "https://www.facebook.com/profile.php?id=100054274566801"
    webbrowser.open(url)

def Whatsapp():
    url = "https://wa.link/438hy8"
    webbrowser.open(url)

def Salir():
    ventana.destroy()


ventana = tkinter.Tk()
ventana.geometry("600x700")
ventana.config(bg="royal blue")
ventana.resizable(0,0)
ventana.title("SEMMAQ DETECTOR VALVULAS")


fondo= tkinter.PhotoImage(file="meow.png")
lbl_imagen = tkinter.Label(ventana, image = fondo, bd=0)
lbl_imagen.place(width=600,height=600 )

logo = tkinter.PhotoImage(file="Logo.png")
lbl_img = tkinter.Label(ventana, image = logo,bd=2, relief="solid")
lbl_img.place(x=10,y=10 )

lbl = Label(ventana, text=" Activador ", bd=2, relief="solid")
lbl.place(x=10,y=130, width=150,height=40)

lbl = Label(ventana, text=" Pagina Web ", bd=2, relief="solid")
lbl.place(x=10,y=190, width=150,height=40)

lbl = Label(ventana, text=" Contactanos ", bd=2, relief="solid")
lbl.place(x=10,y=250, width=150,height=40)

lbl = Label(ventana, text=" Salir ", bd=2, relief="solid")
lbl.place(x=10,y=300, width=150,height=40)

btn = Button(ventana, text="Activar Detector", command=activador, bd=1, relief="solid")
btn.config(fg="blue4",bg="azure")
btn.place(x=180,y=130, width=150,height=40)


btn = Button(ventana, text="Pagina de Facebook", command=Facebook,bd=1, relief="solid")
btn.config(fg="blue4",bg="azure")
btn.place(x=180,y=190, width=150,height=40)


btn = Button(ventana, text="Contacto", command=Contacto,bd=1, relief="solid")
btn.config(fg="blue4",bg="azure")
btn.place(x=180,y=250, width=150,height=40)


btn = Button(ventana, text="Salir de la Pagina", command=Salir,bd=1, relief="solid")
btn.config(fg="blue4",bg="azure")
btn.place(x=180,y=300, width=150,height=40)

numero = tkinter.PhotoImage(file="whatsapp.png").subsample(10, 10)
btn = Button(ventana, image=numero, command=Whatsapp)
btn.config(fg="blue4",bg="white")
btn.place(x=500,y=30, width=70,height=70)

ventana.mainloop()